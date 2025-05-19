import os
import os.path as osp
import PIL.Image as PImage
import numpy as np
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchVAR.utils.data import pil_loader


def get_imagenet_class_mapping(split: str) -> dict:
    """Get ImageNet class name to index mapping."""
    # Load ImageNet class names from torchvision
    train_set = DatasetFolder(
        root=osp.join("./imagenet", split),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=None,
    )
    class_to_idx = train_set.class_to_idx
    return class_to_idx


def save_batch_to_imagenet_structure(
    images, class_labels, start_idx, class_to_idx, output_dir
):
    """
    Save a batch of images in ImageNet-like directory structure.
    Args:
        images: numpy array of shape (B, C, H, W) with values in [0, 255]
        class_labels: numpy array of shape (B,) with class indices
        start_idx: starting index for image naming
        class_to_idx: mapping from class names to indices
        output_dir: base directory to save images
    """
    # Create val directory
    os.makedirs(output_dir, exist_ok=True)

    # Create reverse mapping from index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save images
    for i, (img, label) in enumerate(zip(images, class_labels)):
        # Get class name from index
        class_name = idx_to_class[label]

        # Create class directory if it doesn't exist
        class_dir = osp.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert from (C, H, W) to (H, W, C) and save as JPEG
        img = np.transpose(img, (1, 2, 0))
        img_pil = PImage.fromarray(img)
        img_path = osp.join(class_dir, f"sample_{start_idx + i}.JPEG")
        img_pil.save(img_path, subsampling=0, quality=95)
