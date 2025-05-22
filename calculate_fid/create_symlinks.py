import os
import argparse
import pathlib


def create_symlinks(source_dir):
    """
    Creates symbolic links in a target directory for all files found in source_dir and its subdirectories.
    The target directory is automatically created by appending '_compressed' to the source directory path.
    All symlinks will be created with lowercase extensions to match pytorch-fid requirements.

    Args:
        source_dir (str): The path to the source directory containing image files (can have subfolders).
    """
    source_path = pathlib.Path(source_dir)
    target_dir = str(source_path) + "_compressed"
    target_path = pathlib.Path(target_dir)

    if not source_path.exists() or not source_path.is_dir():
        print(
            f"Error: Source directory '{source_dir}' does not exist or is not a directory."
        )
        return

    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Target directory '{target_dir}' ensured.")

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".JPEG",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    ]  # Add more if needed

    print(f"Scanning '{source_dir}' for image files...")
    files_linked = 0
    files_skipped = 0

    for current_path, _, files in os.walk(source_dir):
        for filename in files:
            source_file_path = pathlib.Path(current_path) / filename
            # Convert extension to lowercase for the target symlink
            target_filename = (
                filename.rsplit(".", 1)[0] + "." + filename.rsplit(".", 1)[1].lower()
            )
            target_symlink_path = target_path / target_filename

            # Check if the file has a common image extension
            if source_file_path.suffix.lower() not in [
                ext.lower() for ext in image_extensions
            ]:
                continue

            try:
                if (
                    not target_symlink_path.exists()
                ):  # Check if symlink or file already exists
                    os.symlink(source_file_path.resolve(), target_symlink_path)
                    files_linked += 1
                else:
                    files_skipped += 1
            except OSError as e:
                print(
                    f"Error creating symlink for {source_file_path} to {target_symlink_path}: {e}"
                )
                files_skipped += 1
            except Exception as e:
                print(f"An unexpected error occurred for {source_file_path}: {e}")
                files_skipped += 1

    print(f"\nProcess complete.")
    print(f"Symbolic links created: {files_linked}")
    print(f"Files skipped (already exist or error): {files_skipped}")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create symbolic links in a target directory for all image files "
        "found in a source directory and its subdirectories. "
        "The target directory is automatically created by appending '_compressed' to the source path."
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Path to the source directory containing image files (can have subfolders).",
    )

    args = parser.parse_args()
    target_dir = create_symlinks(args.source)
    print(f"Creating compressed symlinks in {target_dir}")
