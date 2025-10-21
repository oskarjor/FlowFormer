from absl import app, flags
import json
import torch
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision.transforms.functional import InterpolationMode
from torchVAR.utils.paired_dataset import build_paired_dataset
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
import copy
import time
from torchcfm.utils_SR import (
    generate_samples,
    ema,
    warmup_lr,
    format_time,
    get_unet_params,
    infiniteloop,
)
from torch.amp import GradScaler, autocast
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchVAR.utils.data import denormalize_pm1_into_01

FLAGS = flags.FLAGS

flags.DEFINE_string("json_path", None, help="json path")
flags.DEFINE_string("model_path", None, help="model path")
flags.DEFINE_string("save_dir", None, help="save directory")
flags.DEFINE_string("input_data_path", None, help="input data path")
flags.DEFINE_string("real_data_path", "./imagenet", help="real data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_integer("total_steps", 100000, help="total steps")
flags.DEFINE_string("model", "otcfm", help="model")
flags.DEFINE_integer("print_step", 100, help="print step")
flags.DEFINE_integer("save_step", 10000, help="save step")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay")
flags.DEFINE_float("grad_clip", 1.0, help="gradient clip")
flags.DEFINE_integer("warmup", 5000, help="warmup")
flags.DEFINE_boolean("use_wandb", False, help="use wandb")
flags.DEFINE_boolean("use_amp", False, "Whether to use Automatic Mixed Precision.")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("sigma", 0.0, "sigma")

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

    model_weights = torch.load(FLAGS.model_path, map_location=device)
    net_model.load_state_dict(model_weights["net_model"])
    ema_model = copy.deepcopy(net_model)
    ema_model.load_state_dict(model_weights["ema_model"])
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.learning_rate)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, FLAGS.warmup)
    )

    if FLAGS.use_wandb:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print("Warning: WANDB_API_KEY not found. Wandb logging will be disabled.")
            FLAGS.use_wandb = False
        else:
            try:
                wandb.login(key=api_key)
                run_name = f"SR_finetune_{FLAGS.model}_{json_args['pre_image_size']}_to_{json_args['post_image_size']}_{os.environ.get('SLURM_JOB_ID', 'local')}"

                wandb.init(
                    project="flowformer",
                    entity="oskarjor",
                    name=run_name,
                    config=flags.FLAGS.flag_values_dict(),
                    mode="online" if FLAGS.use_wandb else "disabled",
                )
                # Log model architecture
                wandb.watch(net_model)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                FLAGS.use_wandb = False

    if json_args["naive_upscaling"] == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif json_args["naive_upscaling"] == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {json_args['naive_upscaling']}")

    dataset = build_paired_dataset(
        synthetic_path=FLAGS.input_data_path,
        real_path=FLAGS.real_data_path,
        image_size=json_args["post_image_size"],
        interpolation=upscaling_mode,
        split="train",
    )

    dataloader = DataLoader(
        dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers
    )

    datalooper = infiniteloop(dataloader)

    # LOAD FLOW MATCHER
    sigma = FLAGS.sigma
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

    if FLAGS.use_amp:
        scaler = GradScaler(device=device.type, enabled=FLAGS.use_amp)
    else:
        scaler = None

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Draw 16 random images from the dataset using torch
    num_draw = 16
    # Get dataset length and random indices
    dataset_len = len(dataset)
    rand_indices = torch.randperm(dataset_len)[:num_draw]
    random_x0s = []
    random_x1s = []
    random_ys = []
    for idx in rand_indices:
        x0, x1, y = dataset[idx]
        # If dataset returns tuple (x0, x1, y), just take the input image (e.g., x0)
        random_x0s.append(x0)
        random_x1s.append(x1)
        random_ys.append(y)
    random_x0s = torch.stack(random_x0s, dim=0)
    random_x1s = torch.stack(random_x1s, dim=0)
    random_ys = torch.tensor(random_ys, dtype=torch.int64)
    random_x0s = random_x0s.to(device)
    random_x1s = random_x1s.to(device)
    random_ys = random_ys.to(device)

    for step in range(FLAGS.total_steps):
        optim.zero_grad()

        x0, x1, y = next(datalooper)

        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device)

        with autocast(device_type=device.type, enabled=FLAGS.use_amp):
            t, xt, ut, _, y = FM.guided_sample_location_and_conditional_flow(
                x0, x1, y0=y, y1=y
            )
            vt = net_model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)

        if FLAGS.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
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
                + f"{FLAGS.model}_{json_args['pre_image_size']}_to_{json_args['post_image_size']}_weights_step_{step}_finetuned.pt",
            )

            save_image(
                denormalize_pm1_into_01(random_x0s),
                FLAGS.save_dir + f"x0_step_{step}_samples.png",
                nrow=8,
            )
            save_image(
                denormalize_pm1_into_01(random_x1s),
                FLAGS.save_dir + f"x1_step_{step}_samples.png",
                nrow=8,
            )

            # generate samples
            try:
                generate_samples(
                    net_model,
                    False,
                    FLAGS.save_dir,
                    step,
                    image_size=json_args["post_image_size"],
                    x0=random_x0s,
                    y=random_ys,
                    class_cond=json_args["class_conditional"],
                    num_samples=FLAGS.batch_size,
                    num_classes=NUM_CLASSES,
                    net_="finetuned_net",
                )
            except Exception as e:
                print(
                    f"Warning: Failed to generate samples with net_model at step {step}: {str(e)}"
                )
                print("Continuing training...")

            try:
                generate_samples(
                    ema_model,
                    False,
                    FLAGS.save_dir,
                    step,
                    image_size=json_args["post_image_size"],
                    x0=random_x0s,
                    y=random_ys,
                    class_cond=json_args["class_conditional"],
                    num_samples=FLAGS.batch_size,
                    num_classes=NUM_CLASSES,
                    net_="finetuned_ema",
                )
            except Exception as e:
                print(
                    f"Warning: Failed to generate samples with ema_model at step {step}: {str(e)}"
                )
                print("Continuing training...")


if __name__ == "__main__":
    app.run(finetune_sr)
