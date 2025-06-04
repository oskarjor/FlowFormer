from absl import app, flags
import json
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
import time
from torchcfm.utils_SR import format_time
import os
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("save_dir", None, help="save directory")
flags.DEFINE_string("input_data_path", None, help="input data path")
flags.DEFINE_string("target_data_path", None, help="target data path")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="number of workers")
flags.DEFINE_string("model", "otcfm", help="model")
flags.DEFINE_string("upscaling_mode", "lanczos", help="upscaling mode")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class OTMappedDataset(torch.utils.data.Dataset):
    """
    A dataset that uses pre-computed optimal transport mappings to efficiently
    load corresponding x0 and x1 pairs without storing duplicate images.
    """

    def __init__(self, input_dataset, target_dataset, mapping_file):
        """
        Args:
            input_dataset: The x0 dataset (low resolution)
            target_dataset: The x1 dataset (high resolution)
            mapping_file: Path to the pickle file containing OT mappings
        """
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset

        # Load the optimal transport mappings
        with open(mapping_file, "rb") as f:
            self.ot_mappings = pickle.load(f)

        # Create a flat list of all (x0_idx, x1_idx) pairs from all classes
        self.pairs = []
        for class_idx, class_mappings in self.ot_mappings.items():
            for x0_batch_idx, x1_batch_idx in zip(
                class_mappings["x0_indices"], class_mappings["x1_indices"]
            ):
                # Convert batch indices back to global dataset indices
                x0_global_idx = class_mappings["x0_global_indices"][x0_batch_idx]
                x1_global_idx = class_mappings["x1_global_indices"][x1_batch_idx]
                self.pairs.append((x0_global_idx, x1_global_idx, class_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x0_idx, x1_idx, class_idx = self.pairs[idx]

        # Load the corresponding images
        x0, _ = self.input_dataset[x0_idx]
        x1, _ = self.target_dataset[x1_idx]

        return x0, x1, class_idx


def create_ot_dataset_mappings(argv):
    """Create optimal transport mappings and save them to disk."""
    NUM_CLASSES = 1000
    POST_IMAGE_SIZE = 512

    if FLAGS.upscaling_mode == "nearest":
        upscaling_mode = InterpolationMode.NEAREST
    elif FLAGS.upscaling_mode == "lanczos":
        upscaling_mode = InterpolationMode.LANCZOS
    else:
        raise ValueError(f"Unknown upscaling mode: {FLAGS.upscaling_mode}")

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
        root=osp.join(FLAGS.input_data_path),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=input_transform,
    )
    target_data = DatasetFolder(
        root=osp.join(FLAGS.target_data_path),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=target_transform,
    )

    x0_dataset = SameClassBatchDataset(input_data, NUM_CLASSES)
    x1_dataset = SameClassBatchDataset(target_data, NUM_CLASSES)

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

    # CREATE OT MAPPINGS
    start_time = time.time()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Store mappings for each class
    ot_mappings = {}

    print(f"Creating optimal transport mappings for {NUM_CLASSES} classes...")

    for class_idx in range(NUM_CLASSES):
        print(
            f"{format_time(time.time() - start_time)} | Processing class {class_idx}/{NUM_CLASSES}"
        )

        # Get batch indices for this class from both datasets
        x0_batch_indices = x0_dataset.get_batch_indices(
            FLAGS.batch_size, class_idx=class_idx
        )
        x1_batch_indices = x1_dataset.get_batch_indices(
            FLAGS.batch_size, class_idx=class_idx
        )

        # Load the actual data
        x0_batch = torch.stack([x0_dataset[i][0] for i in x0_batch_indices])
        x1_batch = torch.stack([x1_dataset[i][0] for i in x1_batch_indices])

        x0_batch = x0_batch.to(device)
        x1_batch = x1_batch.to(device)

        # Get the optimal transport plan and indices
        pi = FM.ot_sampler.get_map(x0_batch, x1_batch)
        ot_i, ot_j = FM.ot_sampler.sample_map(pi, FLAGS.batch_size, replace=False)

        # Store the mappings
        ot_mappings[class_idx] = {
            "x0_indices": ot_i.tolist(),  # indices within the batch
            "x1_indices": ot_j.tolist(),  # indices within the batch
            "x0_global_indices": x0_batch_indices,  # global dataset indices
            "x1_global_indices": x1_batch_indices,  # global dataset indices
            "pi": pi.tolist(),  # Store the transport plan as well
        }

    # Save the mappings
    mapping_file = osp.join(FLAGS.save_dir, "ot_mappings.pkl")
    with open(mapping_file, "wb") as f:
        pickle.dump(ot_mappings, f)

    # Save metadata
    metadata = {
        "num_classes": NUM_CLASSES,
        "batch_size": FLAGS.batch_size,
        "model": FLAGS.model,
        "input_data_path": FLAGS.input_data_path,
        "target_data_path": FLAGS.target_data_path,
        "upscaling_mode": FLAGS.upscaling_mode,
        "post_image_size": POST_IMAGE_SIZE,
        "total_pairs": sum(
            len(mappings["x0_indices"]) for mappings in ot_mappings.values()
        ),
    }

    metadata_file = osp.join(FLAGS.save_dir, "ot_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nCompleted optimal transport mapping creation!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Mappings saved to: {mapping_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Total pairs created: {metadata['total_pairs']}")


if __name__ == "__main__":
    app.run(create_ot_dataset_mappings)
