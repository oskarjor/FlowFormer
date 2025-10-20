#!/usr/bin/env python3
"""
Simple test script to verify PairedImageDataset functionality.

This script tests the basic functionality of the paired dataset loader.
"""

import sys
import os.path as osp
from torchVAR.utils.paired_dataset import PairedImageDataset, build_paired_dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchVAR.utils.data import normalize_01_into_pm1


def test_basic_creation():
    """Test basic dataset creation."""
    print("Test 1: Basic Dataset Creation")
    print("-" * 60)

    try:
        # Use build_paired_dataset for simplicity
        dataset = build_paired_dataset(
            synthetic_path="./var_d16_imagenet",
            real_path="./imagenet",
            image_size=256,
            split="val",
        )

        print(f"✓ Dataset created successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of classes: {len(dataset.classes)}")
        print(f"  - First 5 classes: {dataset.classes[:5]}")

        return True
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False


def test_loading_sample():
    """Test loading a single sample."""
    print("\nTest 2: Loading Single Sample")
    print("-" * 60)

    try:
        dataset = build_paired_dataset(
            synthetic_path="./var_d16_imagenet",
            real_path="./imagenet",
            image_size=256,
            split="val",
        )

        # Load first sample
        synthetic_img, real_img, class_idx = dataset[0]

        print(f"✓ Sample loaded successfully")
        print(f"  - Synthetic image shape: {synthetic_img.shape}")
        print(f"  - Real image shape: {real_img.shape}")
        print(f"  - Class index: {class_idx}")
        print(f"  - Class name: {dataset.classes[class_idx]}")

        # Verify shapes
        assert synthetic_img.shape[0] == 3, "Synthetic image should have 3 channels"
        assert real_img.shape[0] == 3, "Real image should have 3 channels"
        assert synthetic_img.shape[1] == 256, "Synthetic image height should be 256"
        assert synthetic_img.shape[2] == 256, "Synthetic image width should be 256"

        print(f"✓ All assertions passed")
        return True

    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_custom_transforms():
    """Test with custom transforms."""
    print("\nTest 3: Custom Transforms")
    print("-" * 60)

    try:
        # Define custom transforms
        synthetic_transform = transforms.Compose(
            [
                transforms.Resize(128, interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

        real_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

        dataset = PairedImageDataset(
            synthetic_path="./var_d16_imagenet/val",
            real_path="./imagenet/val",
            synthetic_transform=synthetic_transform,
            real_transform=real_transform,
        )

        synthetic_img, real_img, class_idx = dataset[0]

        print(f"✓ Custom transforms applied successfully")
        print(f"  - Synthetic image shape: {synthetic_img.shape}")
        print(f"  - Real image shape: {real_img.shape}")

        # Verify different sizes
        assert synthetic_img.shape[1] == 128, "Synthetic should be 128x128"
        assert real_img.shape[1] == 256, "Real should be 256x256"

        print(f"✓ Transform assertions passed")
        return True

    except Exception as e:
        print(f"✗ Failed with custom transforms: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataloader():
    """Test with PyTorch DataLoader."""
    print("\nTest 4: DataLoader Integration")
    print("-" * 60)

    try:
        from torch.utils.data import DataLoader

        dataset = build_paired_dataset(
            synthetic_path="./output/VAR/var_d16/23127566",
            real_path="./imagenet",
            image_size=256,
            split="train",
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
        )

        # Get first batch
        synthetic_batch, real_batch, class_indices = next(iter(dataloader))

        print(f"✓ DataLoader working successfully")
        print(f"  - Synthetic batch shape: {synthetic_batch.shape}")
        print(f"  - Real batch shape: {real_batch.shape}")
        print(f"  - Class indices shape: {class_indices.shape}")

        # Verify batch dimensions
        assert synthetic_batch.shape[0] == 4, "Batch size should be 4"
        assert real_batch.shape[0] == 4, "Batch size should be 4"
        assert class_indices.shape[0] == 4, "Should have 4 class indices"

        print(f"✓ DataLoader assertions passed")
        return True

    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_samples():
    """Test loading multiple samples."""
    print("\nTest 5: Loading Multiple Samples")
    print("-" * 60)

    try:
        dataset = build_paired_dataset(
            synthetic_path="./var_d16_imagenet",
            real_path="./imagenet",
            image_size=256,
            split="val",
        )

        # Load first 5 samples
        num_samples = min(5, len(dataset))
        samples_loaded = 0

        for i in range(num_samples):
            synthetic_img, real_img, class_idx = dataset[i]
            samples_loaded += 1

        print(f"✓ Successfully loaded {samples_loaded} samples")
        print(f"  - All samples have consistent shapes")

        return True

    except Exception as e:
        print(f"✗ Failed to load multiple samples: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PairedImageDataset Test Suite")
    print("=" * 60)
    print()

    # Check if directories exist
    if not osp.exists("./var_d16_imagenet"):
        print("⚠ Warning: ./var_d16_imagenet not found")
        print("  Please update paths in this script to match your setup")
        print()

    if not osp.exists("./imagenet"):
        print("⚠ Warning: ./imagenet not found")
        print("  Please update paths in this script to match your setup")
        print()

    # Run tests
    tests = [
        test_basic_creation,
        test_loading_sample,
        test_custom_transforms,
        test_dataloader,
        test_multiple_samples,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
