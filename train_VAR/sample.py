################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch
import random
import numpy as np
import json
import time
from absl import app, flags
from tqdm import tqdm
from torchVAR.models import build_vae_var
from torchVAR.utils.imagenet_utils import (
    get_imagenet_class_mapping,
    save_batch_to_imagenet_structure,
)

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed


FLAGS = flags.FLAGS

flags.DEFINE_integer("model_depth", 16, help="model depth")
flags.DEFINE_string("vae_ckpt", "vae_ch160v4096z32.pth", help="vae checkpoint")
flags.DEFINE_string("var_ckpt", "var_d16.pth", help="var checkpoint")
flags.DEFINE_string("job_id", None, help="slurm job id")
flags.DEFINE_integer("seed", 0, help="seed")
flags.DEFINE_integer("num_sampling_steps", 250, help="number of sampling steps")
flags.DEFINE_float("cfg", 4.0, help="classifier-free guidance")
flags.DEFINE_integer("num_classes", 1000, help="number of classes in dataset")
flags.DEFINE_bool("more_smooth", False, help="more smooth")
flags.DEFINE_integer("num_samples_per_class", 50, help="number of samples per class")
flags.DEFINE_bool("debug", False, help="debug")
flags.DEFINE_bool("flash_attn", False, help="flash_attn")
flags.DEFINE_bool("fused_mlp", False, help="fused_mlp")
flags.DEFINE_integer("batch_size", 64, help="batch size")
flags.DEFINE_string("split", "val", help="split")
flags.DEFINE_bool("shared_aln", False, help="shared_aln")


def sample_var(argv):
    if FLAGS.job_id == None:
        raise ValueError("job_id is not set")
    output_dir = (
        f"output/VAR/{FLAGS.var_ckpt.split('/')[-1].split('.')[0]}/{FLAGS.job_id}"
    )
    # save flags to flags.json
    os.makedirs(output_dir, exist_ok=True)
    flags_dict = flags.FLAGS.flag_values_dict()
    flags_path = os.path.join(output_dir, "flags.json")
    with open(flags_path, "w") as f:
        json.dump(flags_dict, f, indent=4)

    MODEL_DEPTH = FLAGS.model_depth
    assert MODEL_DEPTH in {16, 20, 24, 30, 36}

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt, var_ckpt = FLAGS.vae_ckpt, FLAGS.var_ckpt
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_ckpt):
        os.system(f"wget {hf_home}/{var_ckpt}")

    # build vae, var
    patch_nums = (
        (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        if MODEL_DEPTH == 36
        else (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "vae" not in globals() or "var" not in globals():
        vae, var = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            device=device,
            patch_nums=patch_nums,
            num_classes=1000,
            depth=MODEL_DEPTH,
            shared_aln=FLAGS.shared_aln,
            flash_if_available=FLAGS.flash_attn,
            fused_if_available=FLAGS.fused_mlp,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    print(f"prepare finished.")

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = FLAGS.seed
    torch.manual_seed(seed)
    num_sampling_steps = (
        FLAGS.num_sampling_steps
    )  # @param {type:"slider", min:0, max:1000, step:1}
    cfg = FLAGS.cfg
    more_smooth = FLAGS.more_smooth
    num_samples_per_class = FLAGS.num_samples_per_class

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    # Get ImageNet class mapping
    class_to_idx = get_imagenet_class_mapping("val")

    # Save class mapping
    mapping_dir = f"./{FLAGS.var_ckpt.split('/')[-1].split('.')[0]}_imagenet"
    os.makedirs(mapping_dir, exist_ok=True)
    with open(osp.join(mapping_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)

    class_labels = [
        i for _ in range(FLAGS.num_samples_per_class) for i in range(FLAGS.num_classes)
    ]
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

    print(f"Sampling {B} images")

    # sample
    start_time = time.time()
    with torch.inference_mode():
        for i in tqdm(range(0, B, FLAGS.batch_size)):
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
            ):  # using bfloat16 can be faster
                current_batch_size = min(FLAGS.batch_size, B - i)
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=current_batch_size,
                    label_B=label_B[i : i + current_batch_size],
                    cfg=cfg,
                    top_k=900,
                    top_p=0.95,
                    g_seed=None,
                    more_smooth=more_smooth,
                )

                images = recon_B3HW.clone().mul_(255).cpu().numpy().astype(np.uint8)
                batch_labels = class_labels[i : i + current_batch_size]

                # Save batch immediately
                save_batch_to_imagenet_structure(
                    images,
                    batch_labels,
                    i,
                    class_to_idx,
                    osp.join(
                        f"./{FLAGS.var_ckpt.split('/')[-1].split('.')[0]}_imagenet",
                        FLAGS.split,
                    ),
                )

    print(f"Sampled {B} images in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    app.run(sample_var)
