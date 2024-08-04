# for colab only
import argparse
import os
import shutil
import pathlib
from google.colab import drive
from joblib import Parallel, delayed


def ensure_root_path(cwd_name: str, target_dir: pathlib.Path, depth=100, __c=1):
    if __c == depth:
        raise Exception("Can't find a unique root directory")
    potential_path = target_dir / f"{cwd_name}_{__c}"
    if potential_path.exists():
        return ensure_root_path(cwd_name, target_dir, depth, __c + 1)
    return potential_path


def copy_item(item, dest):
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OriginalPath", help="Target path in your Google Drive account", type=str,
                        default="/content/drive/MyDrive/projects/forked")
    parser.add_argument("--from_cwd", help="Source path of your directory to be saved", default=True, type=bool, required=False)
    opt = parser.parse_args()

    if not opt.from_cwd:
        raise Exception("Must from cwd!")

    # Mount Google Drive
    # drive.mount('/content/drive')

    # Get the current working directory
    current_directory = pathlib.Path.cwd()

    # Define the target path in Google Drive
    target_path = pathlib.Path(opt.OriginalPath)
    target_path.mkdir(exist_ok=True, parents=True)
    new_root_path = ensure_root_path(current_directory.name, target_path)
    new_root_path.mkdir(exist_ok=True, parents=True)

    # Copy the content from cwd to the target path using joblib for parallel processing
    items = list(current_directory.iterdir())
    Parallel(n_jobs=-1)(delayed(copy_item)(item, new_root_path / item.name) for item in items)

    print(f"Contents of {current_directory} have been copied to {new_root_path}")