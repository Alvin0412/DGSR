# for colab only
import argparse
import shutil
import pathlib
from google.colab import drive


def find_latest_backup(cwd_name: str, target_dir: pathlib.Path, depth=100):
    for i in range(1, depth):
        potential_path = target_dir / f"{cwd_name}_{i}"
        if not potential_path.exists():
            if i == 1:
                raise Exception("No backup directories found")
            return target_dir / f"{cwd_name}_{i - 1}"
    raise Exception("Too many backup directories, cannot determine the latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("OriginalPath", help="Source path in your Google Drive account", type=str)
    parser.add_argument("to_cwd", help="Destination path in your Colab environment", default=True, type=bool)
    opt = parser.parse_args()

    if not opt.to_cwd:
        raise Exception("Must to cwd!")

    # Mount Google Drive
    drive.mount('/content/drive')

    # Get the target path in Google Drive
    target_path = pathlib.Path(opt.OriginalPath)

    # Find the latest backup directory
    current_directory_name = pathlib.Path.cwd().name
    latest_backup_path = find_latest_backup(current_directory_name, target_path)

    # Copy the content from Google Drive to cwd
    for item in latest_backup_path.iterdir():
        dest = pathlib.Path.cwd() / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"Contents of {latest_backup_path} have been copied to {pathlib.Path.cwd()}")