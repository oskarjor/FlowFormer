from absl import app, flags

import torch
import os.path as osp
from torchVAR.utils.data import (
    SameClassBatchDataset,
    pil_loader,
    SameClassBatchDataLoader,
)
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
import os
import torchvision.utils as vutils

FLAGS = flags.FLAGS

flags.DEFINE_string("save_dir", None, help="save directory")
flags.DEFINE_string("input_data_path", None, help="input data path")
flags.DEFINE_string("target_data_path", None, help="target data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_string("model", "otcfm", help="model")
flags.DEFINE_string("naive_upscaling", "nearest", help="naive upscaling mode")
flags.DEFINE_integer("class_idx", 0, help="class index")
flags.DEFINE_integer("num_otcfm_draws", 10, help="number of otcfm draws")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_similar_images(argv):
    NUM_CLASSES = 1000
    POST_IMAGE_SIZE = 512

    if FLAGS.naive_upscaling == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif FLAGS.naive_upscaling == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {FLAGS.naive_upscaling}")

    # LOAD DATASET
    input_transform, target_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    POST_IMAGE_SIZE,
                    interpolation=upscaling_mode,
                ),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(
                    round(POST_IMAGE_SIZE * 1.125),
                    interpolation=InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(POST_IMAGE_SIZE),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        ),
    )
    input_data = DatasetFolder(
        root=osp.join(FLAGS.input_data_path, "train"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )
    target_data = DatasetFolder(
        root=osp.join(FLAGS.target_data_path, "train"),
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

    # x0_datalooper = infiniteloop(x0_dataloader)
    # x1_datalooper = infiniteloop(x1_dataloader)

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

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    x0, y0 = x0_dataloader.get_batch_by_class(class_idx=FLAGS.class_idx)
    for i in range(FLAGS.num_otcfm_draws):
        x1, y1 = x1_dataloader.get_batch_by_class(class_idx=FLAGS.class_idx)

        x0 = x0.to(device)
        x1 = x1.to(device)
        y0 = y0.to(device)
        y1 = y1.to(device)

        # all values in y0 and y1 must be the same
        assert (y0 == y1).all(), "x0 and x1 must have the same class"

        y = y0

        assert y.shape == (FLAGS.batch_size,), "y must be a tensor of (BATCH_SIZE,)"

        # save the image pairs before OT
        save_image_pairs(x0, x1, FLAGS.save_dir, "before_OT.png")

        ot_x0, ot_x1, M = FM.ot_sampler.sample_plan(x0, x1, return_cost=True)

        # save the image pairs after OT
        os.makedirs(osp.join(FLAGS.save_dir, "after_OT"), exist_ok=True)
        save_image_pairs(
            ot_x0,
            ot_x1,
            FLAGS.save_dir,
            f"after_OT_{i}_M_{M.sum().item()}.png",
        )


def save_image_pairs(x0, x1, save_dir, file_name):
    """
    Save image pairs side by side with clear labeling.

    Args:
        x0: Source images tensor (B, C, H, W)
        x1: Target images tensor (B, C, H, W)
        y: Class labels tensor (B,)
        save_dir: Directory to save the images
    """

    # Denormalize images from [-1, 1] to [0, 1]
    x0 = (x0 + 1) / 2
    x1 = (x1 + 1) / 2

    # Save each pair
    for i in range(len(x0)):
        # Create a side-by-side image
        pair = torch.cat([x0[i], x1[i]], dim=2)  # Concatenate horizontally
        save_path = osp.join(save_dir, file_name + f"_{i}.png")
        vutils.save_image(pair, save_path)


if __name__ == "__main__":
    app.run(get_similar_images)
