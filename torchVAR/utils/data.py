import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


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


def build_sr_dataset(
    data_path: str,
    train_reso: int,
    target_reso: int,
    mid_reso=1.125,
    class_indices=None,
):
    # build augmentations
    mid_reso = round(
        mid_reso * target_reso
    )  # first resize to mid_reso, then crop to final_reso
    train_aug, target_train_aug = (
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.BICUBIC
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((target_reso, target_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
            transforms.Resize(train_reso, interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(target_reso, interpolation=InterpolationMode.NEAREST),
        ],
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.BICUBIC
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((target_reso, target_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ],
    )
    target_aug = transforms.Compose(target_train_aug)
    train_aug = transforms.Compose(train_aug)

    val_aug, target_val_aug = (
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.BICUBIC
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((target_reso, target_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
            transforms.Resize(train_reso, interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(target_reso, interpolation=InterpolationMode.NEAREST),
        ],
        [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.BICUBIC
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((target_reso, target_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ],
    )

    # build dataset
    train_set = DatasetFolder(
        root=osp.join(data_path, "train"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_aug,
        target_transform=target_aug,
    )

    val_set = DatasetFolder(
        root=osp.join(data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=val_aug,
        target_transform=target_val_aug,
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
