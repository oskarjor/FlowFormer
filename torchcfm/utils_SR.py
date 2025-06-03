import copy
import os
import time

import torch
from torch import distributed as tdist
from torchdyn.core import NeuralODE
import torchdiffeq

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step, warmup):
    return min(step, warmup) / warmup


def format_time(seconds):
    """Format time in seconds to a human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    tdist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def get_unet_params(unet_conf: str):
    if unet_conf == "normal":
        {
            "num_heads": 8,
            "num_head_channels": 64,
            "attention_resolutions": "16",
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "num_res_blocks": 2,
            "num_channel": 128,
            "groups": 32,
        }
    elif unet_conf == "lightweight":
        return {
            "num_heads": 4,
            "num_head_channels": 16,
            "attention_resolutions": "16",
            "use_scale_shift_norm": True,
            "resblock_updown": True,
            "num_res_blocks": 1,
            "num_channel": 64,
            "groups": 32,
        }
    elif unet_conf == "super_lightweight":
        return {
            "num_heads": 2,
            "num_head_channels": 16,
            "attention_resolutions": "16",
            "use_scale_shift_norm": True,
            "resblock_updown": True,
            "num_res_blocks": 1,
            "num_channel": 32,
            "groups": 8,
        }
    else:
        raise ValueError(f"Unknown unet config: {unet_conf}")


def generate_samples(
    model,
    parallel,
    savedir,
    step,
    time_steps=100,
    image_size=32,
    class_cond=False,
    num_classes=1000,
    net_="normal",
    num_samples=64,
    x0=None,
    y=None,
    compare_samples=False,
    save_png=True,
    method="dopri5",
    atol=1e-4,
    rtol=1e-4,
):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model: The neural network for generating samples
    parallel: bool
        represents the parallel training flag
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    class_cond: bool
        whether to use class conditional generation
    num_classes: int
        number of classes for conditional generation
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)

    with torch.no_grad():
        if x0 is None:
            x0 = torch.randn(num_samples, 3, image_size, image_size, device=device)
        if class_cond:
            if y is None:
                generated_class_list = torch.randint(
                    0, num_classes, (num_samples,), device=device
                )
            else:
                generated_class_list = y

            traj = torchdiffeq.odeint(
                lambda t, x: model_(t, x, generated_class_list),
                x0,
                torch.linspace(0, 1, time_steps, device=device),
                atol=atol,
                rtol=rtol,
                method=method,
            )
        else:
            node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
            traj = node_.trajectory(
                x0,
                t_span=torch.linspace(0, 1, time_steps, device=device),
            )

        # Get final trajectory and post-process
        traj = traj[-1].view([-1, 3, image_size, image_size]).clip(-1, 1)
        traj = traj / 2 + 0.5
    if save_png:
        if not compare_samples:
            save_image(
                traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8
            )
        if compare_samples:
            x0_images = x0.view([-1, 3, image_size, image_size]).clip(-1, 1)
            x0_images = x0_images / 2 + 0.5
            # Create a grid with both x0 and generated images side by side
            comparison = torch.cat([x0_images, traj], dim=0)
            save_image(
                comparison, savedir + f"{net_}_comparison_step_{step}.png", nrow=8
            )
    model.train()
    return traj


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    # values can be either (x0, x1, y) or (x1, y) or (x1)
    while True:
        print("Infiniteloop")
        for values in iter(dataloader):
            yield values


def create_mask(x0, damage_ratio, delta=0.001):
    # Create random mask with damage_ratio of pixels set to 0
    mask = torch.rand_like(x0[:, 0, :, :]) > damage_ratio
    mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    reverse_mask = ~mask
    mask = mask.to(torch.float32)
    reverse_mask = reverse_mask.to(torch.float32)
    reverse_mask = reverse_mask * delta
    mask = mask + reverse_mask
    return mask.to(x0.dtype)
