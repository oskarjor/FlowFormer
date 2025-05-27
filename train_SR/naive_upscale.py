from absl import app, flags
import json
import torch
import os
import numpy as np
import os.path as osp
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torchcfm.utils_SR import generate_samples
from torchcfm.models.unet.unet import UNetModelWrapper
from torchVAR.utils.data import normalize_01_into_pm1, pil_loader
from torchVAR.utils.data import SameClassBatchDataset, SameClassBatchDataLoader
import time
from tqdm import tqdm
from torchVAR.utils.imagenet_utils import (
    save_batch_to_imagenet_structure,
    get_imagenet_class_mapping,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("save_dir", "", help="save directory")
flags.DEFINE_string("data_path", None, help="data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_string("split", "val", help="split")
flags.DEFINE_string("naive_upscaling", "nearest", help="naive upscaling mode")
flags.DEFINE_string("file_format", "png", help="file format")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def sample_sr(argv):
    if FLAGS.naive_upscaling == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif FLAGS.naive_upscaling == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {FLAGS.naive_upscaling}")

    input_transform = transforms.Compose(
        [
            transforms.Resize(
                512,
                interpolation=upscaling_mode,
            ),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )
    input_data = DatasetFolder(
        root=osp.join(FLAGS.data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )

    x0_dataloader = torch.utils.data.DataLoader(
        input_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    class_to_idx = get_imagenet_class_mapping(FLAGS.split)

    start_time = time.time()

    for i, batch in enumerate(x0_dataloader):
        x0, y = batch
        x0 = x0.to(device)
        y = y.to(device)

        images = x0.clone().mul_(255).cpu().numpy().astype(np.uint8)
        class_labels = y.clone().cpu().numpy().astype(np.int32)
        save_batch_to_imagenet_structure(
            images,
            class_labels,
            i * FLAGS.batch_size,
            class_to_idx,
            osp.join(FLAGS.save_dir, FLAGS.split),
            file_format=FLAGS.file_format,
        )
    print(
        f"Sampled {len(x0_dataloader)} images in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    app.run(sample_sr)
