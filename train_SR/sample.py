from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
import shutil
from torchVAR.utils.data import build_SR_dataset, build_npy_dataset
from utils_SR import infiniteloop, generate_samples
from torchcfm.models.unet.unet import UNetModelWrapper
import time

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_string("save_dir", "", help="save directory")
flags.DEFINE_string("data_path", None, help="data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("time_steps", 100, help="time steps")

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

    # build dataset
    dataset = build_npy_dataset(
        data_path=FLAGS.data_path,
        post_image_size=json_args["post_image_size"],
        naive_upscaling=json_args["naive_upscaling"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=json_args["num_workers"],
        drop_last=True,
    )

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

    npy_images = np.zeros(
        (len(dataset), 3, json_args["post_image_size"], json_args["post_image_size"]),
        dtype=np.uint8,
    )
    start_time = time.time()

    for i, (x0, y) in enumerate(dataloader):
        if i % (FLAGS.batch_size * 20) == 0:
            print(
                f"Sampled {i * FLAGS.batch_size} / {len(dataset)} images - {time.time() - start_time:.2f} seconds"
            )
        print(
            f"Sampling {i * FLAGS.batch_size} / {len(dataset)} images - {time.time() - start_time:.2f} seconds"
        )
        print(f"x0: {x0.shape}, y: {y.shape}")
        x0 = x0.to(device)
        y = y.to(device) if json_args["class_conditional"] else None

        print(f"Generating samples for {json_args['batch_size']} images")

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
            num_samples=json_args["batch_size"],
            x0=x0,
            y=y,
            save_png=False,
        )

        images = traj.clone().mul_(255).cpu().numpy().astype(np.uint8)
        npy_images[i * FLAGS.batch_size : (i + 1) * FLAGS.batch_size] = images

    np.save(osp.join(FLAGS.save_dir, "images.npy"), npy_images)
    # copy the class labels from data_path / "class_labels.npy"
    shutil.copy(
        osp.join(FLAGS.data_path, "class_labels.npy"),
        osp.join(FLAGS.save_dir, "class_labels.npy"),
    )

    print(f"Sampled {len(dataset)} images in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    app.run(sample_sr)
