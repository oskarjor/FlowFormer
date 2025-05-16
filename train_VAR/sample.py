################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import json
import time
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm

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
flags.DEFINE_list("return_sizes", [16], help="return sizes")
flags.DEFINE_integer("batch_size", 64, help="batch size")
flags.DEFINE_string("split", "val", help="split")


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def get_imagenet_class_mapping():
    """Get ImageNet class name to index mapping."""
    # Load ImageNet class names from torchvision
    train_set = DatasetFolder(
        root=osp.join("./imagenet", FLAGS.split),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=None,
    )
    class_to_idx = train_set.class_to_idx
    return class_to_idx


def save_batch_to_imagenet_structure(images, class_labels, start_idx, class_to_idx):
    """
    Save a batch of images in ImageNet-like directory structure.
    Args:
        images: numpy array of shape (B, C, H, W) with values in [0, 255]
        class_labels: numpy array of shape (B,) with class indices
        output_dir: base directory to save images
        start_idx: starting index for image naming
        class_to_idx: mapping from class names to indices
    """
    # Create val directory
    output_dir = osp.join(
        f"./{FLAGS.var_ckpt.split('/')[-1].split('.')[0]}_imagenet", FLAGS.split
    )
    os.makedirs(output_dir, exist_ok=True)

    # Create reverse mapping from index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save images
    for i, (img, label) in enumerate(zip(images, class_labels)):
        # Get class name from index
        class_name = idx_to_class[label]

        # Create class directory if it doesn't exist
        class_dir = osp.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert from (C, H, W) to (H, W, C) and save as JPEG
        img = np.transpose(img, (1, 2, 0))
        img_pil = PImage.fromarray(img)
        img_path = osp.join(class_dir, f"sample_{start_idx + i:05d}.JPEG")
        img_pil.save(img_path, subsampling=0, quality=95)


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
    return_sizes = [int(x) for x in FLAGS.return_sizes]

    assert all(x in patch_nums for x in return_sizes), (
        f"return_sizes must be a subset of patch_nums: {patch_nums}"
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
            shared_aln=False,
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
    class_to_idx = get_imagenet_class_mapping()

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
                    g_seed=seed,
                    more_smooth=more_smooth,
                )

                images = recon_B3HW.clone().mul_(255).cpu().numpy().astype(np.uint8)
                batch_labels = class_labels[i : i + current_batch_size]

                # Save batch immediately
                save_batch_to_imagenet_structure(images, batch_labels, i, class_to_idx)

    print(f"Sampled {B} images in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    app.run(sample_var)
