from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from utils_SR import generate_samples
from torchcfm.models.unet.unet import UNetModelWrapper
from torchVAR.utils.data import normalize_01_into_pm1, pil_loader
from torchVAR.utils.data import SameClassBatchDataset, SameClassBatchDataLoader
import time
from tqdm import tqdm
from torchVAR.utils.imagenet_utils import (
    save_batch_to_imagenet_structure,
    get_imagenet_class_mapping,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_string("save_dir", "", help="save directory")
flags.DEFINE_string("data_path", None, help="data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("time_steps", 100, help="time steps")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_string("split", "val", help="split")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def read_json_flags(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def sample_sr(argv):
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
    net_model.eval()

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save flags to json file
    flags_dict = flags.FLAGS.flag_values_dict()
    flags_path = os.path.join(FLAGS.save_dir, "flags.json")
    with open(flags_path, "w") as f:
        json.dump(flags_dict, f, indent=4)

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
        root=osp.join(FLAGS.data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )

    x0_dataset = SameClassBatchDataset(input_data, NUM_CLASSES)

    x0_dataloader = SameClassBatchDataLoader(
        x0_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    class_to_idx = get_imagenet_class_mapping(FLAGS.split)

    start_time = time.time()

    for i in range(len(x0_dataloader)):
        x0, y = next(x0_dataloader)
        x0 = x0.to(device)
        y = y.to(device) if json_args["class_conditional"] else None

        traj = generate_samples(
            net_model,
            parallel=False,
            savedir=FLAGS.save_dir,
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

        images = traj.clone().mul_(255).cpu().numpy().astype(np.uint8)
        class_labels = y.clone().cpu().numpy().astype(np.int32)
        save_batch_to_imagenet_structure(
            images,
            class_labels,
            i,
            class_to_idx,
            osp.join(FLAGS.save_dir, FLAGS.split),
        )
    print(
        f"Sampled {len(x0_dataloader)} images in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    app.run(sample_sr)
