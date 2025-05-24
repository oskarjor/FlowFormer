# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import time
import json

import torch
from absl import app, flags
from torchcfm.utils_SR import ema, generate_samples, infiniteloop, warmup_lr, format_time

from torchVAR.utils.data import build_SR_dataset

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/SR/", help="output_directory")
flags.DEFINE_string("save_dir", None, help="save_directory")

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
flags.DEFINE_integer("pre_image_size", 32, help="image size")
flags.DEFINE_integer("post_image_size", 64, help="image size")
flags.DEFINE_string("dataset", "imagenet", help="dataset")
flags.DEFINE_list("class_indices", None, help="class indices")
flags.DEFINE_bool("debug", False, help="debug mode")
flags.DEFINE_bool("use_amp", False, help="whether to use automatic mixed precision")
flags.DEFINE_string("naive_upscaling", "nearest", help="naive upscaling method")

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

# wandb
flags.DEFINE_bool("use_wandb", True, help="whether to use wandb logging")
flags.DEFINE_string("wandb_project", "flowformer", help="wandb project name")
flags.DEFINE_string("wandb_entity", None, help="wandb entity/username")
flags.DEFINE_string("wandb_name", None, help="wandb run name")

# validation
flags.DEFINE_integer("validation_steps", 100, help="number of validation steps")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
        FLAGS.save_dir = f"./results/SR/{FLAGS.model}/"

    if FLAGS.class_indices is not None:
        FLAGS.class_indices = [int(i) for i in FLAGS.class_indices]

    # DATASETS/DATALOADER
    if FLAGS.dataset == "imagenet":
        if FLAGS.post_image_size not in [32, 64, 128, 256, 512]:
            raise ValueError(
                "Imagenet only supports 32x32, 64x64, 128x128, 256x256, 512x512 images"
            )
        num_classes = 1000
        train_set, val_set = build_SR_dataset(
            data_path="./imagenet",
            pre_image_size=FLAGS.pre_image_size,
            post_image_size=FLAGS.post_image_size,
            class_indices=FLAGS.class_indices,
            naive_upscaling=FLAGS.naive_upscaling,
        )

        if FLAGS.debug:
            print(f"Train set: {train_set.class_to_idx}")
    else:
        raise ValueError(f"Unknown dataset {FLAGS.dataset}")

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    datalooper = infiniteloop(dataloader)

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_datalooper = infiniteloop(val_dataloader)

    # MODELS
    if FLAGS.pre_image_size == 32 and FLAGS.post_image_size == 64:
        num_heads = 4
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 2
        num_channel = 128

    elif FLAGS.pre_image_size == 256 and FLAGS.post_image_size == 512:
        num_heads = 8
        num_head_channels = 64
        attention_resolutions = "16"
        use_scale_shift_norm = True
        resblock_updown = False
        num_res_blocks = 2
        num_channel = 128
    else:
        raise ValueError(
            f"Unknown image size: {FLAGS.pre_image_size}->{FLAGS.post_image_size}"
        )

    net_model = UNetModelWrapper(
        dim=(3, FLAGS.post_image_size, FLAGS.post_image_size),
        num_res_blocks=num_res_blocks,
        num_channels=num_channel,
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
        use_fp16=FLAGS.use_amp,
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
                    or f"SR_{FLAGS.model}_{FLAGS.pre_image_size}_to_{FLAGS.post_image_size}_{os.environ.get('SLURM_JOB_ID', 'local')}"
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
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, FLAGS.warmup)
    )
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
        if FLAGS.debug:
            print(f"Step {step} of {FLAGS.total_steps}")

        optim.zero_grad()

        x0, x1, y = next(datalooper)
        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device) if FLAGS.class_conditional else None

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

            # Validate on a batch from validation set
            net_model.eval()
            with torch.no_grad():
                val_loss = 0
                recon_loss = 0
                for val_step in range(FLAGS.validation_steps):
                    val_x0, val_x1, val_y = next(val_datalooper)
                    val_x0, val_x1 = val_x0.to(device), val_x1.to(device)
                    val_y = val_y.to(device) if FLAGS.class_conditional else None

                    val_t, val_xt, val_ut = FM.sample_location_and_conditional_flow(
                        val_x0, val_x1
                    )
                    val_vt = net_model(val_t, val_xt, val_y)
                    val_loss = torch.mean((val_vt - val_ut) ** 2)

                    if val_step == 0:
                        # generate samples
                        generated_x1 = generate_samples(
                            net_model,
                            FLAGS.parallel,
                            FLAGS.save_dir,
                            step,
                            image_size=FLAGS.post_image_size,
                            x0=val_x0,
                            y=val_y,
                            class_cond=FLAGS.class_conditional,
                            num_samples=FLAGS.batch_size,
                            num_classes=num_classes,
                        )
                        net_model.eval()
                        # calculate the reconstruction loss
                        recon_loss = torch.mean((generated_x1 - val_x1) ** 2)
                        print(f"Reconstruction Loss: {recon_loss.item():.4f}")

                    val_loss += val_loss.item()

                val_loss /= FLAGS.validation_steps

                print(f"Validation Loss: {val_loss.item():.4f}")

                if FLAGS.use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/step": step,
                            "val/recon_loss": recon_loss,
                        }
                    )

            net_model.train()


def demo(argv):
    print("DEMO")
    train_set, _ = build_SR_dataset(
        data_path="./imagenet",
        pre_image_size=FLAGS.pre_image_size,
        post_image_size=FLAGS.post_image_size,
    )

    first_sample, first_target, _ = train_set[0]
    print(first_sample.shape, first_target.shape)

    print("PLOTTING...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(
        first_sample.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
    )  # Denormalize from [-1,1] to [0,1]
    plt.title("Input (Low Resolution)")
    plt.subplot(1, 2, 2)
    plt.imshow(first_target.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5)
    plt.title("Target (High Resolution)")
    print("SAVING...")
    plt.savefig("demo.png")


def check_model_size(argv):
    num_heads = 4
    num_head_channels = 64
    attention_resolutions = "16"
    use_scale_shift_norm = True
    resblock_updown = False
    num_res_blocks = 2
    num_channel = 128
    num_classes = 1000

    net_model = UNetModelWrapper(
        dim=(3, FLAGS.post_image_size, FLAGS.post_image_size),
        num_res_blocks=num_res_blocks,
        num_channels=num_channel,
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
        use_fp16=FLAGS.use_amp,
    ).to(device)

    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))


if __name__ == "__main__":
    # app.run(check_model_size)
    app.run(train)
