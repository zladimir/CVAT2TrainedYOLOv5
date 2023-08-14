from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Result:
    image: str
    width: int
    height: int
    clazz: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class AnnotationsParser:
    def __init__(self, source: str, relative_path: bool=False) -> None:
        self.source = Path(source)
        self.relative_path = relative_path
        self.annotations = []

    def parse(self):
        for ann_path in self.source.glob('**/*.xml'):
            content = get_content(ann_path)
            for image in content.find_all('image'):
                image_annotation = image_processing(image, ann_path.parent)
                self.annotations.extend(image_annotation)
        self.annotations = pd.DataFrame(self.annotations)
        if self.relative_path:
            self.annotations['image'] = self.annotations['image'].apply(
                lambda x: x.relative_to(self.source))
        return self.annotations


def get_content(path: Path) -> bs:
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
    return bs_content


def image_processing(image: bs, path: Path) -> pd.DataFrame:
    image_folder = [f for f in path.iterdir() if f.is_dir()]
    if len(image_folder) > 1:
        raise ValueError(f'Multiple image directories at {path}')
    image_abs_name = path.joinpath(*image_folder, image.attrs['name'])
    width = int(image.attrs['width'])
    height = int(image.attrs['height'])
    polygons = image.find_all('polygon')
    labels = []
    for polygon in polygons:
        bbox = polygon_processing(polygon)
        labels.append(Result(image_abs_name, width, height, *bbox))
    return labels


def polygon_processing(polygon: bs):
    points = polygon.attrs['points']
    # TODO: refactoring
    points = np.array(
        [list(map(float, point.split(','))) for point in points.split(';')]
    ).astype(int)
    clazz = polygon.find('attribute').text
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    return (clazz, x_min, y_min, x_max, y_max)
