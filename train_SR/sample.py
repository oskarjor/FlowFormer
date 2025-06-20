from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
from torchcfm.utils_SR import generate_samples, get_unet_params
from torchcfm.models.unet.unet import UNetModelWrapper
import time
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
flags.DEFINE_integer("num_samples", 10000, help="number of samples")
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
        groups=unet_params["groups"],
    ).to(device)

    # load model
    model_weights = torch.load(FLAGS.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])
    net_model.eval()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    start_time = time.time()

    class_to_idx = get_imagenet_class_mapping(FLAGS.split)

    print("Sampling...")

    num_samples_per_class = FLAGS.num_samples // NUM_CLASSES

    y_values = [[i] * num_samples_per_class for i in range(NUM_CLASSES)]
    y_values = np.concatenate(y_values)

    for i in range(FLAGS.num_samples // FLAGS.batch_size + 1):
        print(f"Sampling {i * FLAGS.batch_size} / {FLAGS.num_samples} images")

        y = torch.from_numpy(
            y_values[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size]
        ).to(device)

        current_batch_size = min(FLAGS.batch_size, y.shape[0])

        x0 = torch.randn(
            current_batch_size,
            3,
            json_args["post_image_size"],
            json_args["post_image_size"],
        ).to(device)

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
            num_samples=current_batch_size,
            x0=x0,
            y=y,
            save_png=False,
        )

        images = traj.clone().mul_(255).cpu().numpy().astype(np.uint8)
        class_labels = y.clone().cpu().numpy().astype(np.uint8)
        save_batch_to_imagenet_structure(
            images,
            class_labels,
            i,
            class_to_idx,
            osp.join(FLAGS.save_dir, FLAGS.split),
        )

    print(
        f"Sampled {FLAGS.num_samples} images in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    app.run(sample_sr)
