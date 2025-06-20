from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torchcfm.utils_SR import generate_samples, get_unet_params
from torchcfm.models.unet.unet import UNetModelWrapper, QKVAttention, QKVAttentionLegacy
from torchVAR.utils.data import normalize_01_into_pm1, pil_loader

# Try to import thop, fallback to fvcore if not available
import thop

USE_THOP = True
print("Using thop for FLOP calculation")

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("time_steps", 20, help="time steps")
flags.DEFINE_integer("num_workers", 4, help="number of workers")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def read_json_flags(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def calculate_flops_thop(model, t, x, y=None):
    """Calculate FLOPs using thop package with custom ops for attention."""
    custom_ops = {
        QKVAttention: QKVAttention.count_flops,
        QKVAttentionLegacy: QKVAttentionLegacy.count_flops,
    }

    if y is not None:
        inputs = (t, x, y)
    else:
        inputs = (t, x)

    try:
        macs, params = thop.profile(
            model, inputs=inputs, custom_ops=custom_ops, verbose=False
        )
        # Convert MACs to FLOPs (1 MAC = 2 FLOPs for multiply-accumulate)
        flops = macs * 2
        return flops
    except Exception as e:
        print(f"thop FLOP calculation failed: {e}")
        return None


def sample_sr(argv):
    NUM_CLASSES = 1000

    json_path = FLAGS.json_path
    if json_path is None:
        raise ValueError("json_path is required")
    json_args = read_json_flags(json_path)

    # MODELS
    unet_params = get_unet_params(json_args["unet_conf"])

    print("Loading model...")
    net_model = UNetModelWrapper(
        dim=(3, json_args["post_image_size"], json_args["post_image_size"]),
        num_res_blocks=unet_params["num_res_blocks"],
        num_channels=unet_params["num_channel"],
        channel_mult=None,
        num_heads=unet_params["num_heads"],
        num_head_channels=unet_params["num_head_channels"],
        attention_resolutions=unet_params["attention_resolutions"],
        use_scale_shift_norm=unet_params["use_scale_shift_norm"],
        resblock_updown=unet_params["resblock_updown"],
        dropout=0.1,
        class_cond=json_args["class_conditional"],
        num_classes=NUM_CLASSES,
        use_new_attention_order=True,
        use_fp16=False,
        use_checkpoint=False,  # Disable checkpointing for FLOP analysis
        groups=unet_params["groups"],
    ).to(device)

    # load model
    model_weights = torch.load(FLAGS.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])
    net_model.eval()

    # build dataset
    if json_args["naive_upscaling"] == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif json_args["naive_upscaling"] == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {json_args['naive_upscaling']}")

    input_transform = transforms.Compose(
        [
            transforms.Resize(
                json_args["post_image_size"],
                interpolation=upscaling_mode,
            ),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )
    input_data = DatasetFolder(
        root=osp.join("var_d16_imagenet", "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )

    x0_dataloader = torch.utils.data.DataLoader(
        input_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    x0, y = next(iter(x0_dataloader))
    x0 = x0.to(device)
    y = y.to(device) if json_args["class_conditional"] else None

    # Count FLOPs for a single forward pass
    print("Calculating FLOPs...")
    t = torch.tensor([0.0], device=device)

    # Use torch.no_grad() to avoid any gradient computation during FLOP analysis
    with torch.no_grad():
        flops_per_step = None

        print("Trying thop FLOP calculation...")
        flops_per_step = calculate_flops_thop(net_model, t, x0, y)
        print(
            f"TFLOPs per model evaluation: {round(flops_per_step / 1_000_000_000_000, 3)}"
        )

    print("Generating samples...")
    traj = generate_samples(
        net_model,
        parallel=False,
        savedir=None,
        step=json_args["total_steps"],
        time_steps=FLAGS.time_steps,
        image_size=json_args["post_image_size"],
        class_cond=json_args["class_conditional"],
        num_classes=NUM_CLASSES,
        net_="normal",
        num_samples=FLAGS.batch_size,
        x0=x0,
        y=y,
        save_png=False,
    )


if __name__ == "__main__":
    app.run(sample_sr)
