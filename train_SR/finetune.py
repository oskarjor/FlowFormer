from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
import shutil
from torchVAR.utils.data import (
    SameClassBatchDataset,
    build_dataset,
    pil_loader,
    SameClassBatchDataLoader,
)
from utils_SR import infiniteloop, generate_samples
from torchcfm.models.unet.unet import UNetModelWrapper
import time
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchVAR.utils.data import normalize_01_into_pm1
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_string("save_dir", None, help="save directory")
flags.DEFINE_string("input_data_path", None, help="input data path")
flags.DEFINE_string("target_data_path", None, help="target data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def read_json_flags(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def finetune_sr(argv):
    NUM_CLASSES = 1000

    json_path = FLAGS.json_path
    if json_path is None:
        raise ValueError("json_path is required")
    if FLAGS.save_dir is None:
        raise ValueError("save_dir is required")
    json_args = read_json_flags(json_path)

    # MODELS
    if json_args["pre_image_size"] == 32 and json_args["post_image_size"] == 64:
        num_heads = 4
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 2
        num_channel = 128

    elif json_args["pre_image_size"] == 256 and json_args["post_image_size"] == 512:
        num_heads = 8
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 2
        num_channel = 128
    else:
        raise ValueError(
            f"Unknown image size: {json_args['pre_image_size']}->{json_args['post_image_size']}"
        )

    print("Loading model...")
    net_model = UNetModelWrapper(
        dim=(3, json_args["post_image_size"], json_args["post_image_size"]),
        num_res_blocks=num_res_blocks,
        num_channels=num_channel,
        channel_mult=None,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        dropout=0.1,
        class_cond=json_args["class_conditional"],
        num_classes=NUM_CLASSES,
        use_new_attention_order=True,
        use_fp16=False,
    ).to(device)

    # load model
    model_weights = torch.load(FLAGS.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])

    if json_args["naive_upscaling"] == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif json_args["naive_upscaling"] == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {json_args['naive_upscaling']}")

    # finetune
    input_transform, target_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    json_args["post_image_size"],
                    interpolation=upscaling_mode,
                ),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(
                    round(json_args["post_image_size"] * 1.125),
                    interpolation=InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(json_args["post_image_size"]),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        ),
    )
    input_data = DatasetFolder(
        root=osp.join(FLAGS.input_data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )
    target_data = DatasetFolder(
        root=osp.join(FLAGS.target_data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=target_transform,
    )

    x0_dataset = SameClassBatchDataset(input_data, NUM_CLASSES)
    x1_dataset = SameClassBatchDataset(target_data, NUM_CLASSES)

    x0_dataloader = SameClassBatchDataLoader(
        x0_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    x1_dataloader = SameClassBatchDataLoader(
        x1_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    for _ in range(10):
        class_idx = np.random.randint(0, NUM_CLASSES)
        x0, y0 = next(x0_dataloader, class_idx)
        x1, y1 = next(x1_dataloader, class_idx)
        print(x0.shape, x1.shape)
        print(y0, y1)
        break


if __name__ == "__main__":
    app.run(finetune_sr)
