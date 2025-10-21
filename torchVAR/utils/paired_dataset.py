"""
Dataset for loading paired synthetic and real ImageNet images.

This module provides a dataset class that loads pairs of images where:
- Synthetic images (subset) are stored as .png files
- Real ImageNet images are stored as .JPEG files
- Both follow the same directory structure (e.g., train/n01440764/image_name.ext)
"""

import os.path as osp
from typing import Optional, Callable
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms.functional import InterpolationMode


def pil_loader(path):
    """Load an image from disk and convert to RGB."""
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


class PairedImageDataset(DatasetFolder):
    """
    A dataset that loads pairs of synthetic and real images.

    The synthetic images determine which images to load (subset of all real images).
    For each synthetic image (.png), the corresponding real image (.JPEG) is loaded
    from the real dataset path by matching the directory structure and filename.

    Args:
        synthetic_path: Root path to synthetic images (e.g., './synthetic_imagenet/train')
        real_path: Root path to real ImageNet images (e.g., './imagenet/train')
        synthetic_transform: Optional transform to apply to synthetic images
        real_transform: Optional transform to apply to real images
        loader: Function to load images (default: pil_loader)
        synthetic_extensions: Tuple of valid extensions for synthetic images (default: ('.png',))
        real_extension: Extension for real images (default: '.JPEG')

    Returns:
        A tuple of (synthetic_image, real_image, class_idx) where:
        - synthetic_image: The loaded synthetic image (after transform if provided)
        - real_image: The loaded real image (after transform if provided)
        - class_idx: The class index (integer)

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize(256),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = PairedImageDataset(
        ...     synthetic_path='./var_d16_imagenet/train',
        ...     real_path='./imagenet/train',
        ...     synthetic_transform=transform,
        ...     real_transform=transform
        ... )
        >>> synthetic_img, real_img, class_idx = dataset[0]
    """

    def __init__(
        self,
        synthetic_path: str,
        real_path: str,
        synthetic_transform: Optional[Callable] = None,
        real_transform: Optional[Callable] = None,
        loader: Callable = pil_loader,
        synthetic_extensions: tuple = (".png",),
        real_extension: str = ".JPEG",
    ):
        # Initialize the base DatasetFolder with synthetic path
        # This will find all synthetic images and create the class structure
        super().__init__(
            root=synthetic_path,
            loader=loader,
            extensions=synthetic_extensions,
            transform=None,  # We'll handle transforms manually
        )

        self.synthetic_path = synthetic_path
        self.real_path = real_path
        self.synthetic_transform = synthetic_transform
        self.real_transform = real_transform
        self.synthetic_extensions = synthetic_extensions
        self.real_extension = real_extension

        # Verify that real_path exists
        if not osp.exists(real_path):
            raise ValueError(f"Real image path does not exist: {real_path}")

        print(f"[PairedImageDataset] Loaded {len(self.samples)} paired samples")
        print(f"[PairedImageDataset] Synthetic path: {synthetic_path}")
        print(f"[PairedImageDataset] Real path: {real_path}")
        print(f"[PairedImageDataset] Number of classes: {len(self.classes)}")

    def _get_real_image_path(self, synthetic_path: str) -> str:
        """
        Convert a synthetic image path to the corresponding real image path.

        Args:
            synthetic_path: Full path to synthetic image

        Returns:
            Full path to corresponding real image

        Example:
            If synthetic_path = '/path/to/synthetic/train/n01440764/image_0001.png'
            and real_path = '/path/to/real'
            Returns: '/path/to/real/train/n01440764/image_0001.JPEG'
        """
        # Get the relative path from the synthetic root
        rel_path = osp.relpath(synthetic_path, self.synthetic_path)

        # Get the directory and filename
        dir_path, filename = osp.split(rel_path)

        # Change the extension to the real image extension
        filename_without_ext = osp.splitext(filename)[0]
        real_filename = filename_without_ext + self.real_extension

        # Construct the full real image path
        real_path = osp.join(self.real_path, dir_path, real_filename)

        return real_path

    def __getitem__(self, index: int):
        """
        Get a pair of synthetic and real images.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (synthetic_image, real_image, class_idx)
        """
        # Get the synthetic image path and class
        synthetic_img_path, class_idx = self.samples[index]

        # Load synthetic image
        synthetic_img = self.loader(synthetic_img_path)

        # Get corresponding real image path
        real_img_path = self._get_real_image_path(synthetic_img_path)

        # Check if real image exists
        if not osp.exists(real_img_path):
            raise FileNotFoundError(
                f"Real image not found: {real_img_path}\n"
                f"Expected to pair with synthetic: {synthetic_img_path}"
            )

        # Load real image
        real_img = self.loader(real_img_path)

        # Apply transforms if provided
        if self.synthetic_transform is not None:
            synthetic_img = self.synthetic_transform(synthetic_img)

        if self.real_transform is not None:
            real_img = self.real_transform(real_img)

        return synthetic_img, real_img, class_idx


def build_paired_dataset(
    synthetic_path: str,
    real_path: str,
    image_size: int,
    interpolation: InterpolationMode = InterpolationMode.LANCZOS,
    synthetic_transform: Optional[Callable] = None,
    real_transform: Optional[Callable] = None,
    split: str = "train",
):
    """
    Build a paired dataset with default transforms.

    Args:
        synthetic_path: Root path to synthetic images
        real_path: Root path to real images
        image_size: Target image size for resizing
        synthetic_transform: Optional custom transform for synthetic images
        real_transform: Optional custom transform for real images
        split: Dataset split ('train' or 'val')

    Returns:
        PairedImageDataset instance
    """
    from torchvision import transforms
    from torchVAR.utils.data import normalize_01_into_pm1

    # Default transforms if not provided
    if synthetic_transform is None:
        synthetic_transform = transforms.Compose(
            [
                transforms.Resize(
                    round(image_size * 1.125), interpolation=interpolation
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

    if real_transform is None:
        real_transform = transforms.Compose(
            [
                transforms.Resize(
                    round(image_size * 1.125), interpolation=interpolation
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

    # Build dataset
    dataset = PairedImageDataset(
        synthetic_path=osp.join(synthetic_path, split),
        real_path=osp.join(real_path, split),
        synthetic_transform=synthetic_transform,
        real_transform=real_transform,
    )

    return dataset
