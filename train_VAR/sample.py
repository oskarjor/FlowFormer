################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
from torchVAR.models import VQVAE, build_vae_var

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("model_depth", 16, help="model depth")
flags.DEFINE_string("vae_ckpt", "vae_ch160v4096z32.pth", help="vae checkpoint")
flags.DEFINE_string("var_ckpt", "var_d16.pth", help="var checkpoint")
flags.DEFINE_string("output_dir", "./results/VAR/", help="output_directory")
flags.DEFINE_integer("seed", 0, help="seed")
flags.DEFINE_integer("num_sampling_steps", 250, help="number of sampling steps")
flags.DEFINE_float("cfg", 4.0, help="classifier-free guidance")
flags.DEFINE_list(
    "class_labels", [980, 980, 437, 437, 22, 22, 562, 562], help="class labels"
)
flags.DEFINE_bool("more_smooth", False, help="more smooth")
flags.DEFINE_integer("num_samples_per_class", 50, help="number of samples per class")
flags.DEFINE_bool("debug", False, help="debug")


def sample_var(argv):
    MODEL_DEPTH = FLAGS.model_depth
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt, var_ckpt = FLAGS.vae_ckpt, FLAGS.var_ckpt
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_ckpt):
        os.system(f"wget {hf_home}/{var_ckpt}")

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
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
            shared_aln=False,
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
    class_labels = FLAGS.class_labels
    more_smooth = FLAGS.more_smooth
    num_samples_per_class = FLAGS.num_samples_per_class

    if FLAGS.debug:
        print(class_labels)
        print(type(class_labels))
    
    class_labels = [int(x) for x in class_labels]

    if FLAGS.debug:
        print(class_labels)
        print(type(class_labels))

    expanded_class_labels = [
        x for x in class_labels for _ in range(num_samples_per_class)
    ]

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

    # sample
    B = len(expanded_class_labels)
    label_B: torch.LongTensor = torch.tensor(expanded_class_labels, device=device)
    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):  # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B,
                label_B=label_B,
                cfg=cfg,
                top_k=900,
                top_p=0.95,
                g_seed=seed,
                more_smooth=more_smooth,
            )

    # save images
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    for i, recon_B3HW in enumerate(recon_B3HW):
        PImage.fromarray(recon_B3HW.mul_(255).cpu().numpy().astype(np.uint8)).save(
            osp.join(
                FLAGS.output_dir,
                f"class_{expanded_class_labels[i]}_image_{i % num_samples_per_class}.png",
            )
        )


if __name__ == "__main__":
    app.run(sample_var)
