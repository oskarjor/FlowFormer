import copy
import os

import torch
from torch import distributed as tdist
from torchdyn.core import NeuralODE
import torchdiffeq

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        if class_cond:
            # Generate random class labels
            generated_class_list = torch.randint(
                0, num_classes, (num_samples,), device=device
            )
            # Set first 16 samples to same class (340 = zebra)
            # generated_class_list[:16] = torch.tensor([340], device=device)

            # Use torchdiffeq's odeint with class conditioning
            traj = torchdiffeq.odeint(
                lambda t, x: model_(t, x, generated_class_list),
                torch.randn(num_samples, 3, image_size, image_size, device=device),
                torch.linspace(0, 1, time_steps, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        else:
            traj = node_.trajectory(
                torch.randn(num_samples, 3, image_size, image_size, device=device),
                t_span=torch.linspace(0, 1, time_steps, device=device),
            )

        # Get final trajectory and post-process
        traj = traj[-1].view([-1, 3, image_size, image_size]).clip(-1, 1)
        traj = traj / 2 + 0.5

    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y
