from absl import app, flags
import json
import torch
import os

from torchVAR.utils.data import build_SR_dataset
from utils_SR import infiniteloop, generate_samples
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_integer("num_batches", 1, help="number of batches")
flags.DEFINE_string("save_dir", "", help="save directory")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def read_json_flags(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def sample_sr(argv):
    json_path = FLAGS.json_path
    if json_path is None:
        raise ValueError("json_path is required")
    if FLAGS.save_dir is None:
        raise ValueError("save_dir is required")
    json_args = read_json_flags(json_path)

    # build dataset
    dataset = build_SR_dataset(
        data_path=json_args["data_path"],
        pre_image_size=json_args["pre_image_size"],
        post_image_size=json_args["post_image_size"],
        class_indices=json_args["class_indices"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=json_args["batch_size"],
        shuffle=True,
        num_workers=json_args["num_workers"],
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    if json_args["pre_image_size"] == 32 and json_args["post_image_size"] == 64:
        num_heads = 4
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
        num_classes=json_args["num_classes"],
        use_new_attention_order=True,
        use_fp16=False,
    ).to(device)

    # load model
    model_weights = torch.load(flags.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])
    net_model.eval()

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for batch_idx in range(FLAGS.num_batches):
        x0, x1, y = next(datalooper)
        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device) if FLAGS.class_conditional else None

        generate_samples(
            net_model,
            parallel=False,
            savedir=FLAGS.save_dir,
            step=json_args["total_steps"],
            time_steps=100,
            image_size=json_args["post_image_size"],
            class_cond=json_args["class_conditional"],
            num_classes=json_args["num_classes"],
            net_="normal",
            num_samples=json_args["batch_size"],
            x0=x0,
            y=y,
        )


if __name__ == "__main__":
    app.run(sample_sr)
