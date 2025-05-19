import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader


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
        # Ensure data is in the correct format and type at initialization
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
        # Store data in (H, W, C) format to avoid transposition during __getitem__
        if data.shape[1] == 3:  # If data is in (N, C, H, W) format
            self.data = np.transpose(data, (0, 2, 3, 1))
        else:
            self.data = data
        self.class_labels = class_labels.astype(np.int64)  # Convert labels once
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # Already in (H, W, C) format
        if self.transform is not None:
            img = PImage.fromarray(img).convert("RGB")
            img = self.transform(img)
        return img, self.class_labels[idx]


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
    data = np.load(osp.join(data_path, "images.npy"), mmap_mode="r")
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


class SameClassBatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        A wrapper dataset that ensures all images in a batch come from the same class.
        Takes advantage of ImageNet's folder structure where each class is in its own folder.
        Args:
            dataset: The base dataset (should be an ImageNet dataset)
        """
        self.dataset = dataset
        # Get the class folders and their corresponding indices
        self.class_folders = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_batch_indices(self, batch_size, class_idx=None):
        """
        Get indices for a batch where all images are from the same class.
        Args:
            batch_size: Size of the batch to generate
        Returns:
            List of indices for the batch
        """
        if class_idx is None:
            # Randomly select a class folder
            class_folder = np.random.choice(self.class_folders)
            class_idx = self.class_to_idx[class_folder]

        # Get all samples for this class
        class_samples = [
            i for i, (_, c) in enumerate(self.dataset.samples) if c == class_idx
        ]

        # Randomly select batch_size samples from this class
        if len(class_samples) < batch_size:
            # If we don't have enough samples, sample with replacement
            batch_indices = np.random.choice(
                class_samples, size=batch_size, replace=True
            )
        else:
            # Otherwise sample without replacement
            batch_indices = np.random.choice(
                class_samples, size=batch_size, replace=False
            )
        return batch_indices.tolist()


class SameClassBatchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=0):
        super().__init__(dataset, batch_size, num_workers)

    def __iter__(self):
        return super().__iter__()

    def __next__(self, class_idx=None):
        batch_indices = self.dataset.get_batch_indices(
            self.batch_size, class_idx=class_idx
        )
        batch = [self.dataset[i] for i in batch_indices]
        return batch
