# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import time
import json

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
flags.DEFINE_string("dataset", "imagenet", help="dataset")
flags.DEFINE_list("class_indices", None, help="class indices")
flags.DEFINE_bool("debug", False, help="debug mode")

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

# Add these with your other flags in train.py
flags.DEFINE_bool("use_wandb", True, help="whether to use wandb logging")
flags.DEFINE_string("wandb_project", "flowformer", help="wandb project name")
flags.DEFINE_string("wandb_entity", None, help="wandb entity/username")
flags.DEFINE_string("wandb_name", None, help="wandb run name")

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

    if FLAGS.class_indices is not None:
        FLAGS.class_indices = [int(i) for i in FLAGS.class_indices]

    # DATASETS/DATALOADER
    if FLAGS.dataset == "cifar10":
        if FLAGS.image_size != 32:
            raise ValueError("CIFAR10 only supports 32x32 images")
        num_classes = 10
        train_set = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif FLAGS.dataset == "imagenet":
        if FLAGS.image_size not in [32, 64, 128, 256]:
            raise ValueError(
                "Imagenet only supports 32x32, 64x64, 128x128, 256x256 images"
            )
        num_classes, train_set, val_set = build_dataset(
            data_path="./imagenet",
            final_reso=FLAGS.image_size,
            hflip=True,
            mid_reso=1.125,
            class_indices=FLAGS.class_indices,
        )
    elif FLAGS.dataset == "lsun":
        if FLAGS.image_size not in [32, 64]:
            raise ValueError("LSUN only supports 32x32 or 64x64 images")
        classes = [
            "bedroom_train"
        ]  # Other classes: "church_outdoor_train", "classroom_train", "dining_room_train", "kitchen_train", "living_room_train"
        num_classes = len(classes)
        train_set = datasets.LSUN(
            root="./data",
            classes=classes,
            transform=transforms.Compose(
                [
                    transforms.Resize(FLAGS.image_size),
                    transforms.CenterCrop(FLAGS.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown dataset {FLAGS.dataset}")

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
        num_heads = 4
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 3
    elif FLAGS.image_size == 32:
        num_heads = 4
        num_head_channels = 32
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 2
    elif FLAGS.image_size == 128:
        num_heads = 4
        num_head_channels = 128
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 3
    elif FLAGS.image_size == 256:
        num_heads = 4
        num_head_channels = 256
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 3
    else:
        raise ValueError(f"Unknown image size {FLAGS.image_size}")

    net_model = UNetModelWrapper(
        dim=(3, FLAGS.image_size, FLAGS.image_size),
        num_res_blocks=num_res_blocks,
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
        use_new_attention_order=True,
        use_fp16=False,
    ).to(device)

    if FLAGS.use_wandb:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print("Warning: WANDB_API_KEY not found. Wandb logging will be disabled.")
            FLAGS.use_wandb = False
        else:
            try:
                wandb.login(key=api_key)
                run_name = (
                    FLAGS.wandb_name
                    or f"cfm_{FLAGS.model}_{FLAGS.image_size}x{FLAGS.image_size}_{os.environ.get('SLURM_JOB_ID', 'local')}"
                )

                wandb.init(
                    project=FLAGS.wandb_project,
                    entity=FLAGS.wandb_entity,
                    name=run_name,
                    config=flags.FLAGS.flag_values_dict(),
                    mode="online" if FLAGS.use_wandb else "disabled",
                )
                # Log model architecture
                wandb.watch(net_model)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                FLAGS.use_wandb = False

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

    # Save flags to json file
    flags_dict = flags.FLAGS.flag_values_dict()
    flags_path = os.path.join(FLAGS.save_dir, "flags.json")
    with open(flags_path, "w") as f:
        json.dump(flags_dict, f, indent=4)

    print(f"Saved flags to {flags_path}")

    for step in range(FLAGS.total_steps):
        optim.zero_grad()
        x1, y = next(datalooper)
        x1 = x1.to(device)
        y = y.to(device) if FLAGS.class_conditional else None
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        if FLAGS.debug:
            print("Parameters before forward pass:")
            print(f"t: {t.shape} \n xt: {xt.shape} \n ut: {ut.shape} \n y: {y.shape}")
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
            normal_samples = generate_samples(
                net_model,
                FLAGS.parallel,
                FLAGS.save_dir,
                step,
                time_steps=100,
                class_cond=FLAGS.class_conditional,
                num_classes=num_classes,
                net_="normal",
            )
            ema_samples = generate_samples(
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
                FLAGS.save_dir
                + f"{FLAGS.model}_{FLAGS.image_size}x{FLAGS.image_size}_weights_step_{step}.pt",
            )


if __name__ == "__main__":
    app.run(train)
