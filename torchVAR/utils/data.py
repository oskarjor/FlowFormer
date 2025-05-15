import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import numpy as np
import torch


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
    class_indices=None,
):
    # build augmentations
    mid_reso = round(
        mid_reso * final_reso
    )  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = (
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ],
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ],
    )
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

    # build dataset
    train_set = DatasetFolder(
        root=osp.join(data_path, "train"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_aug,
    )
    val_set = DatasetFolder(
        root=osp.join(data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=val_aug,
    )

    # Filter classes if indices are provided
    if class_indices is not None:
        print(f"Filtering classes: {class_indices}")
        # Get the class names in order
        idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}
        selected_classes = [idx_to_class[i] for i in class_indices]
        print(f"Selecting train classes: {selected_classes}")

        # Create new class mappings
        new_class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        new_classes = selected_classes

        # Filter samples and remap class indices
        train_set.samples = [
            (p, new_class_to_idx[train_set.classes[c]])
            for p, c in train_set.samples
            if train_set.classes[c] in selected_classes
        ]
        val_set.samples = [
            (p, new_class_to_idx[val_set.classes[c]])
            for p, c in val_set.samples
            if val_set.classes[c] in selected_classes
        ]

        # Update class mappings
        train_set.class_to_idx = new_class_to_idx
        train_set.classes = new_classes
        val_set.class_to_idx = new_class_to_idx
        val_set.classes = new_classes

        print(f"Filtered train set: {len(train_set.samples)}")
        print(f"Filtered val set: {len(val_set.samples)}")
        num_classes = len(class_indices)
    else:
        num_classes = 1000

    print(f"[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}")
    print_aug(train_aug, "[train]")
    print_aug(val_aug, "[val]")

    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def print_aug(transform, label):
    print(f"Transform {label} = ")
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print("---------------------------\n")


class SR_DatasetFolder(DatasetFolder):
    def __init__(self, pre_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_transform = pre_transform

    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        sample = self.loader(path)
        sample = self.pre_transform(sample)
        if self.transform is not None:
            image = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(sample)

        return image, target, class_idx


# Create custom dataset class
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, class_labels, transform=None):
        self.data = data
        self.class_labels = class_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # convert it to a PIL image
        img = img.astype(np.uint8)
        print("IMG SHAPE", img.shape)
        img = PImage.fromarray(img).convert("RGB")
        print("IMG SIZE", img.size)
        # Convert from (H, W, C) to (C, H, W) if needed
        if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_labels[idx].astype(np.int64)


def build_npy_dataset(
    data_path: str,
    post_image_size: int,
    naive_upscaling="nearest",
):
    """
    Build a dataset from a numpy file containing images.
    Args:
        data_path: Path to the numpy file containing images
        naive_upscaling: The naive upscaling method to use
    """

    if naive_upscaling == "nearest":
        interpolation = InterpolationMode.NEAREST
    elif naive_upscaling == "bicubic":
        interpolation = InterpolationMode.BICUBIC
    elif naive_upscaling == "lanczos":
        interpolation = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Invalid naive_upscaling method {naive_upscaling}")

    transform = transforms.Compose(
        [
            transforms.Resize(
                post_image_size, interpolation=interpolation
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    # Load the numpy file
    data = np.load(osp.join(data_path, "images.npy"))
    class_labels = np.load(osp.join(data_path, "class_labels.npy"))
    if len(data.shape) != 4:
        raise ValueError(
            f"Expected numpy array of shape (N, H, W, C), got {data.shape}"
        )

    dataset = NumpyDataset(data, class_labels, transform=transform)
    print(f"[Dataset] {len(dataset)=}")
    print(f"[Class labels] {len(class_labels)=}")
    return dataset


def build_SR_dataset(
    data_path: str,
    pre_image_size: int,
    post_image_size: int,
    hflip=False,
    mid_reso=1.125,
    naive_upscaling="nearest",
):
    assert naive_upscaling in ["nearest", "bicubic", "lanczos"], (
        f"Invalid naive_upscaling method {naive_upscaling}"
    )

    if naive_upscaling == "nearest":
        interpolation = InterpolationMode.NEAREST
    elif naive_upscaling == "bicubic":
        interpolation = InterpolationMode.BICUBIC
    elif naive_upscaling == "lanczos":
        interpolation = InterpolationMode.LANCZOS

    # build augmentations
    mid_reso = round(
        mid_reso * post_image_size
    )  # first resize to mid_reso, then crop to final_reso

    # Pre-transforms to get the high-res image
    train_pre_aug, val_pre_aug = (
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((post_image_size, post_image_size)),
        ],
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((post_image_size, post_image_size)),
        ],
    )

    if hflip:
        train_pre_aug.insert(0, transforms.RandomHorizontalFlip())
    train_pre_aug = transforms.Compose(train_pre_aug)
    val_pre_aug = transforms.Compose(val_pre_aug)

    # Transforms to create the degraded version
    # First resize down to low-res, then back up to high-res
    degrade_transform = transforms.Compose(
        [
            transforms.Resize(
                pre_image_size, interpolation=InterpolationMode.LANCZOS
            ),  # Downscale
            transforms.Resize(
                post_image_size, interpolation=interpolation
            ),  # Upscale back
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    # Transform for the high-res target
    target_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    train_set = SR_DatasetFolder(
        root=osp.join(data_path, "train"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        pre_transform=train_pre_aug,
        transform=degrade_transform,
        target_transform=target_transform,
    )

    val_set = SR_DatasetFolder(
        root=osp.join(data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        pre_transform=val_pre_aug,
        transform=degrade_transform,
        target_transform=target_transform,
    )

    return train_set, val_set
