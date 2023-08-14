from PIL import Image
import yaml
import pandas as pd
import shutil
from pathlib import Path


def read_image(path: str) -> Image:
    return Image.open(path)


def read_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def image_copy(source: Path, destination: Path):
    """Copy image from source to destination"""
    Path.mkdir(destination.parent, parents=True, exist_ok=True)
    shutil.copy(source, destination)


def save_csv(table: pd.DataFrame, path: str):
    path = Path(path)
    Path.mkdir(path.parent, parents=True, exist_ok=True)
    table.to_csv(path)
