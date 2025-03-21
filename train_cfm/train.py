# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import time

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from utils_cifar import ema, generate_samples, infiniteloop

from torchVAR.utils.data import build_dataset

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
flags.DEFINE_string("save_dir", None, help="save_directory")

# UNet
flags.DEFINE_integer("num_channel", 256, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 800001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_integer("image_size", 32, help="image size")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "print_step",
    1000,
    help="frequency of printing training progress",
)
flags.DEFINE_bool("class_conditional", False, help="class conditional training")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def format_time(seconds):
    """Format time in seconds to a human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    start_time = time.time()

    if FLAGS.save_dir is None:
        FLAGS.save_dir = f"./results/{FLAGS.model}/"

    # DATASETS/DATALOADER
    # dataset = datasets.CIFAR10(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     ),
    # )

    num_classes, train_set, val_set = build_dataset(
        data_path="./imagenet",
        final_reso=FLAGS.image_size,
        hflip=True,
        mid_reso=1.125,
    )

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    if FLAGS.image_size == 64:
        num_heads = 8
        num_head_channels = 32
        attention_resolutions = "8"
        use_scale_shift_norm = True
        resblock_updown = True
    else:
        num_heads = 4
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = True

    net_model = UNetModelWrapper(
        dim=(3, FLAGS.image_size, FLAGS.image_size),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=None,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        dropout=0.1,
        class_cond=FLAGS.class_conditional,
        num_classes=num_classes,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

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

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for step in range(FLAGS.total_steps):
        optim.zero_grad()
        x1, y = next(datalooper)
        x1 = x1.to(device)
        y = y.to(device) if FLAGS.class_conditional else None
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = net_model(t, xt, y)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)  # new

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

        # sample and Saving the weights
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
            generate_samples(
                net_model,
                FLAGS.parallel,
                FLAGS.save_dir,
                step,
                time_steps=100,
                class_cond=FLAGS.class_conditional,
                num_classes=num_classes,
                net_="normal",
            )
            generate_samples(
                ema_model,
                FLAGS.parallel,
                FLAGS.save_dir,
                step,
                time_steps=100,
                class_cond=FLAGS.class_conditional,
                num_classes=num_classes,
                net_="ema",
            )
            torch.save(
                {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                },
                FLAGS.save_dir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
            )


if __name__ == "__main__":
    app.run(train)
