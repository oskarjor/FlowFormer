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
        # Get the class names in order
        class_names = sorted(train_set.classes)
        # Filter to only include specified classes
        selected_classes = [class_names[i] for i in class_indices]

        # Filter train set
        train_indices = [
            i
            for i, (_, class_idx) in enumerate(train_set.samples)
            if train_set.classes[class_idx] in selected_classes
        ]
        train_set.samples = [train_set.samples[i] for i in train_indices]
        train_set.targets = [train_set.targets[i] for i in train_indices]

        # Filter val set
        val_indices = [
            i
            for i, (_, class_idx) in enumerate(val_set.samples)
            if val_set.classes[class_idx] in selected_classes
        ]
        val_set.samples = [val_set.samples[i] for i in val_indices]
        val_set.targets = [val_set.targets[i] for i in val_indices]

        # Update number of classes
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
