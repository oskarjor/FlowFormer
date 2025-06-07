"""
Robust training augmentations for Flow Matching models to handle VAR-generated images.
These augmentations simulate the kinds of artifacts and variations that VAR models typically produce.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import random


class VARRobustAugmentation:
    """
    Augmentation strategy designed to make models robust to VAR-generated image characteristics.
    """

    def __init__(self, prob=0.7):
        self.prob = prob

    def quantization_noise(self, img, levels=256, prob=0.3):
        """Simulate vector quantization artifacts from VAR."""
        if random.random() < prob:
            # Reduce bit depth to simulate quantization
            quantized = torch.round(img * (levels - 1)) / (levels - 1)
            # Add slight noise to simulate reconstruction errors
            noise = torch.randn_like(img) * 0.01
            return torch.clamp(quantized + noise, 0, 1)
        return img

    def frequency_distortion(self, img, prob=0.25):
        """Simulate the frequency characteristics of autoregressive generation."""
        if random.random() < prob:
            # Convert to frequency domain
            img_np = img.permute(1, 2, 0).numpy()
            fft = np.fft.fft2(img_np, axes=(0, 1))

            # Slightly dampen high frequencies (VAR tends to be smoother)
            h, w = fft.shape[:2]
            damping = np.ones((h, w, 3))
            center_h, center_w = h // 2, w // 2

            # Create frequency damping mask
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - center_h) ** 2 + (j - center_w) ** 2)
                    if dist > min(h, w) * 0.3:  # High frequency
                        damping[i, j] *= 0.8 + 0.4 * random.random()

            fft_modified = fft * damping
            img_modified = np.real(np.fft.ifft2(fft_modified, axes=(0, 1)))
            return torch.clamp(torch.from_numpy(img_modified).permute(2, 0, 1), 0, 1)
        return img

    def autoregressive_patterns(self, img, prob=0.2):
        """Simulate subtle directional artifacts from autoregressive generation."""
        if random.random() < prob:
            h, w = img.shape[-2:]

            # Create subtle raster-scan bias
            raster_bias = torch.zeros_like(img)
            for i in range(h):
                for j in range(w):
                    # Slight bias based on raster position
                    bias_strength = 0.02 * (i * w + j) / (h * w)
                    raster_bias[:, i, j] = bias_strength * (random.random() - 0.5)

            return torch.clamp(img + raster_bias, 0, 1)
        return img

    def color_statistics_shift(self, img, prob=0.4):
        """Simulate the different color statistics of synthetic images."""
        if random.random() < prob:
            # Random color channel scaling
            scales = torch.normal(1.0, 0.1, (3, 1, 1))
            scales = torch.clamp(scales, 0.8, 1.2)

            # Random color shifts
            shifts = torch.normal(0.0, 0.05, (3, 1, 1))

            return torch.clamp(img * scales + shifts, 0, 1)
        return img

    def compression_artifacts(self, img, prob=0.3):
        """Simulate compression-like artifacts from token-based generation."""
        if random.random() < prob:
            # Convert to PIL for JPEG-like compression simulation
            img_pil = transforms.ToPILImage()(img)

            # Add slight blur to simulate lossy compression
            blur_radius = random.uniform(0.2, 0.8)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            return transforms.ToTensor()(img_pil)
        return img

    def texture_smoothing(self, img, prob=0.3):
        """Simulate the tendency of VAR to produce smoother textures."""
        if random.random() < prob:
            # Slight gaussian blur with probability
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.3, 0.7)

            blur_transform = transforms.GaussianBlur(kernel_size, sigma)
            blurred = blur_transform(img)

            # Mix original and blurred
            mix_ratio = random.uniform(0.1, 0.3)
            return (1 - mix_ratio) * img + mix_ratio * blurred
        return img

    def __call__(self, img):
        """Apply augmentations with overall probability."""
        if random.random() > self.prob:
            return img

        # Apply augmentations in random order
        augmentations = [
            self.quantization_noise,
            self.color_statistics_shift,
            self.compression_artifacts,
            self.texture_smoothing,
            self.frequency_distortion,
            self.autoregressive_patterns,
        ]

        # Randomly apply 1-3 augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)

        for aug in selected_augs:
            img = aug(img)

        return img


class StandardRobustAugmentation:
    """
    Standard domain robustness augmentations.
    """

    def __init__(self, prob=0.8):
        self.prob = prob

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

        self.blur_transforms = [
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            transforms.GaussianBlur(5, sigma=(0.1, 1.5)),
        ]

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Color augmentations
        if random.random() < 0.6:
            img = self.color_jitter(img)

        # Blur augmentations
        if random.random() < 0.3:
            blur_transform = random.choice(self.blur_transforms)
            img = blur_transform(img)

        # Noise augmentation
        if random.random() < 0.4:
            noise = torch.randn_like(img) * random.uniform(0.01, 0.05)
            img = torch.clamp(img + noise, 0, 1)

        return img


def create_robust_training_transforms(image_size, augment_prob=0.7):
    """
    Create training transforms with VAR-robustness augmentations.

    Args:
        image_size: Target image size
        augment_prob: Probability of applying augmentations
    """

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.125)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # Apply VAR-specific augmentations to input images only
            VARRobustAugmentation(prob=augment_prob),
            # Standard augmentations
            StandardRobustAugmentation(prob=augment_prob * 0.8),
            # Normalize to [-1, 1] if needed
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_target_transforms(image_size):
    """
    Create transforms for target images (minimal augmentation).
    """
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.125)),
            transforms.CenterCrop(image_size),  # Use center crop for targets
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
