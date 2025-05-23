from absl import app, flags
import json
import torch
import numpy as np
import os.path as osp
from torchVAR.utils.data import (
    SameClassBatchDataset,
    pil_loader,
    SameClassBatchDataLoader,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchVAR.utils.data import normalize_01_into_pm1
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
import copy
import time
import wandb
from utils_SR import generate_samples, ema, warmup_lr, format_time, infiniteloop

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_string("save_dir", None, help="save directory")
flags.DEFINE_string("input_data_path", None, help="input data path")
flags.DEFINE_string("target_data_path", None, help="target data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_integer("total_steps", 100000, help="total steps")
flags.DEFINE_string("model", "otcfm", help="model")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def read_json_flags(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def finetune_sr(argv):
    NUM_CLASSES = 1000

    # READ JSON FLAGS OF PRETRAINED MODEL
    json_path = FLAGS.json_path
    if json_path is None:
        raise ValueError("json_path is required")
    if FLAGS.save_dir is None:
        raise ValueError("save_dir is required")
    json_args = read_json_flags(json_path)

    # LOAD PRETRAINED UNET MODEL
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

    model_weights = torch.load(FLAGS.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])
    ema_model = copy.deepcopy(net_model)
    ema_model.load_state_dict(model_weights["ema_model"])
    optim = torch.optim.Adam(net_model.parameters(), lr=json_args["lr"])
    optim.load_state_dict(model_weights["optim"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, json_args["warmup"])
    )
    sched.load_state_dict(model_weights["sched"])

    if json_args["naive_upscaling"] == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif json_args["naive_upscaling"] == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {json_args['naive_upscaling']}")

    # LOAD DATASET
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

    x0_datalooper = infiniteloop(x0_dataloader)
    x1_datalooper = infiniteloop(x1_dataloader)

    # LOAD FLOW MATCHER
    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    # FINETUNE
    start_time = time.time()
    for step in range(FLAGS.total_steps):
        optim.zero_grad()

        x0, y0 = next(x0_datalooper)
        x1, y1 = next(x1_datalooper)

        x0 = x0.to(device)
        x1 = x1.to(device)
        y0 = y0.to(device)
        y1 = y1.to(device)

        assert y0 == y1, "x0 and x1 must have the same class"

        y = y0

        assert y.shape == (FLAGS.batch_size, 1), "y must be a tensor of (BATCH_SIZE, 1)"

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = net_model(t, xt, y)
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)

        # Print training progress at intervals
        if step % FLAGS.print_step == 0:
            current_lr = optim.param_groups[0]["lr"]
            elapsed_time = time.time() - start_time
            steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = FLAGS.total_steps - step
            estimated_remaining_time = (
                remaining_steps / steps_per_second if steps_per_second > 0 else 0
            )

            print(
                f"Step {step}/{FLAGS.total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {format_time(elapsed_time)} | "
                f"ETA: {format_time(estimated_remaining_time)}"
            )

            if FLAGS.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/step": step,
                        "train/elapsed_time": elapsed_time,
                        "train/eta": estimated_remaining_time,
                    }
                )

        # sample and Saving the weights
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
            # Save the model
            torch.save(
                {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                FLAGS.save_dir
                + f"{FLAGS.model}_{FLAGS.pre_image_size}_to_{FLAGS.post_image_size}_weights_step_{step}.pt",
            )

            # generate samples
            generate_samples(
                net_model,
                False,
                FLAGS.save_dir,
                step,
                image_size=json_args["post_image_size"],
                x0=x0,
                y=y,
                class_cond=json_args["class_conditional"],
                num_samples=FLAGS.batch_size,
                num_classes=NUM_CLASSES,
                net_="finetuned_net",
            )

            generate_samples(
                ema_model,
                False,
                FLAGS.save_dir,
                step,
                image_size=json_args["post_image_size"],
                x0=x0,
                y=y,
                class_cond=json_args["class_conditional"],
                num_samples=FLAGS.batch_size,
                num_classes=NUM_CLASSES,
                net_="finetuned_ema",
            )


if __name__ == "__main__":
    app.run(finetune_sr)
