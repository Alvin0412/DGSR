# for colab only
import argparse
import shutil
import pathlib
from google.colab import drive
from joblib import Parallel, delayed


def find_latest_backup(cwd_name: str, target_dir: pathlib.Path):
    potential_path = target_dir / f"{cwd_name}"
    if not potential_path.exists():
        raise Exception("No backup directories found")
    return potential_path


def copy_item(item, dest):
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OriginalPath", help="Source path in your Google Drive account", type=str)
    parser.add_argument("--to_cwd", help="Destination path in your Colab environment", default=False, type=bool)
    parser.add_argument("--directory_name", help="Destination path in your Colab environment", default=True, type=str)
    opt = parser.parse_args()

    # Mount Google Drive
    # drive.mount('/content/drive')

    # Get the target path in Google Drive
    target_path = pathlib.Path(opt.OriginalPath)
    if opt.to_cwd and opt.directory_name:
        raise Exception("can't both")
    # Find the latest backup directory
    if opt.to_cwd:
        directory_name = pathlib.Path.cwd().name
    else:
        directory_name = opt.directory_name
    latest_backup_path = find_latest_backup(directory_name, target_path)

    # Copy the content from Google Drive to cwd using joblib for parallel processing
    items = list(latest_backup_path.iterdir())
    Parallel(n_jobs=-1)(delayed(copy_item)(item, pathlib.Path.cwd() / item.name) for item in items)

    print(f"Contents of {latest_backup_path} have been copied to {pathlib.Path.cwd()}")