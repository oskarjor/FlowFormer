import os
from PIL import Image
import random
import math
from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string("folder_path", None, "Path to the folder containing images")
flags.DEFINE_integer(
    "grid_size", None, "Size of the grid (e.g., 8 for an 8x8 grid = 64 images)"
)


def get_random_images(folder_path, n_images):
    """
    Get n random images from a folder and its subfolders.
    Returns a list of PIL Image objects.
    """
    image_files = []

    # Walk through all subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, file))

    if len(image_files) < n_images:
        raise ValueError(
            f"Not enough images found. Need {n_images} images but found only {len(image_files)}"
        )

    # Randomly select n images
    selected_files = random.sample(image_files, n_images)
    return [Image.open(f) for f in selected_files]


def create_image_grid(images, grid_size):
    """
    Create a grid of images.
    Assumes all images are square and of the same size.
    """
    if not images:
        return None

    # Get image size (all images are assumed to be the same size)
    img_size = images[0].size[0]  # width and height are the same

    # Create a new image with a white background
    grid_width = img_size * grid_size
    grid_height = img_size * grid_size
    grid_image = Image.new("RGB", (grid_width, grid_height), "white")

    # Place images in the grid
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size

        # Calculate position (no need for centering since images are the same size)
        x = col * img_size
        y = row * img_size

        grid_image.paste(img, (x, y))

    return grid_image


def main(argv):
    # Validate required flags
    if not FLAGS.folder_path:
        raise ValueError("--folder_path is required")
    if not FLAGS.grid_size:
        raise ValueError("--grid_size is required")

    # Calculate number of images needed
    n_images = FLAGS.grid_size * FLAGS.grid_size

    # Get random images
    images = get_random_images(FLAGS.folder_path, n_images)

    # Create grid
    grid_image = create_image_grid(images, FLAGS.grid_size)

    # Generate output filename from the entire path
    path_string = FLAGS.folder_path.replace("/", "_").replace("\\", "_")
    output_filename = (
        f"images/{path_string}_grid_{FLAGS.grid_size}x{FLAGS.grid_size}.jpg"
    )

    # Save the result
    grid_image.save(output_filename)
    print(f"Grid image saved as {output_filename}")


if __name__ == "__main__":
    app.run(main)
