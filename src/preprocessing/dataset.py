import pandas as pd
import numpy as np
from typing import Dict, Tuple, Callable
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.utils import image_copy, read_image
from src.utils.logging import create_logger
from src.utils.constants import YOLO_COLUMNS


class BoxNormalizator:
    def __init__(self, pos2class: dict):
        self.pos2class = pos2class

    def normalize(self, labels: pd.DataFrame) -> pd.DataFrame:
        labels['clazz'] = labels['clazz'].map(self.pos2class)
        labels = coords_normalization(labels)
        return labels


class YoloFormatter:
    def __init__(self, class2pos: dict, source: str, yolo_dir: str):
        self.class2pos = class2pos
        self.source = source
        self.yolo_dir = yolo_dir

    def save_data(self, image_saver: Callable[[Path, Path], None],
                  box_getter: Callable[[pd.DataFrame, Path], np.ndarray],
                  image_names: Path, labels: pd.DataFrame, sample: str,
                  suffix: str) -> None:
        """Сохранение изображений и аннотаций под формат Yolov5"""
        for image in image_names:
            image_name, ann_name = get_instance_names(image, suffix)
            image_src = Path(self.source, image)
            image_dst = Path(self.yolo_dir, sample, image_name)
            image_saver(image_src, image_dst)
            ann_dst = Path(self.yolo_dir, sample, ann_name)
            bboxes = box_getter(labels, image)
            save_annotations(bboxes, ann_dst)

    def create_yaml(self):
        """Создание yaml файла"""
        yaml_path = Path(self.yolo_dir, 'dataset.yaml')
        data = get_yolo_info(self.yolo_dir, self.class2pos)
        with open(yaml_path, 'w') as f:
            f.write(f'{data["train"]}\n{data["val"]}\n')
            f.write(f'{data["nc"]}\n{data["names"]}')


class Dataset:
    def __init__(self, cfg: dict):
        self.logger = create_logger()

        self.yolo_dir = Path(cfg['dataset']['yolo_dataset'])
        self.source = Path(cfg['source'])
        self.valid_size = cfg['train_test_split']['valid_size']
        self.random_state = cfg['train_test_split']['random_state']
        self.lr_flip = cfg['model']['lr_flip']

        # Normalization of rectangle coordinates
        self.normalizator = BoxNormalizator(cfg['classes']['pos2class'])

        # Formatting a dataset for Yolov5
        self.formatter = YoloFormatter(cfg['classes']['class2pos'],
                                       cfg['source'],
                                       cfg['dataset']['yolo_dataset'])

    def make(self, labels: pd.DataFrame, lr_suffix: str='_flip'):
        labels = self.normalizator.normalize(labels)
        data = train_val_split(labels, self.valid_size, self.random_state)
        self.logger.info(f"train_images: {data['train'].shape[0]}")
        self.logger.info(f"valid_images: {data['valid'].shape[0]}")

        make_dataset_struct(self.yolo_dir)
        for sample in ('train', 'valid'):
            image_names = data[sample]
            self.formatter.save_data(image_copy, get_bboxes, image_names,
                                     labels, sample, suffix=None)
            if (sample == 'train') and self.lr_flip:
                self.formatter.save_data(make_lr_flip, get_lr_flipped_bboxes,
                                         image_names, labels, sample,
                                         suffix=lr_suffix)
        self.formatter.create_yaml()


def get_lr_flip(box_str: str) -> list:
    """LR Отображние боксов"""
    box = [float(value) for value in box_str.split(' ')]
    box[1] = 1.0 - box[1]
    box[0] = get_class_lr_flip(int(box[0]))
    return box


def get_image_name(source: Path, image: Path) -> Path:
    pattern = str(image.with_suffix('')) + '.*'
    extension = [
        ext.suffix for ext in source.glob(pattern) if ext.suffix != '.txt'
    ]
    if len(extension) != 1:
        raise RuntimeError('Impossible to identify an image by annotation')
    return image.with_suffix(extension[0])


def lr_flip_txt(source: Path, image: Path) -> list:
    """Чтение аннотации и LR flip"""
    bboxes = []
    with open(Path(source, image), 'r') as fin:
        for line in fin:
            lr_box = get_lr_flip(line.rstrip())
            image_name = get_image_name(source, image)
            bboxes.append([image_name, *lr_box])
    return bboxes


def get_lr_labels(source: Path) -> pd.DataFrame:
    """Формирование LR аннотаций"""
    labels = []
    for annotation in source.glob('*.txt'):
        annotation = Path(annotation.name)
        labels.extend(lr_flip_txt(source, annotation))
    return pd.DataFrame(labels, columns=YOLO_COLUMNS)


def check_flip_presence(source: Path, suffix: str) -> bool:
    for annotation in source.glob('*.txt'):
        if suffix in annotation.name:
            return True
    return False


def flip_yolo_dataset(cfg: dict) -> None:
    """LR flip собранного датасета YOLO из внешнего источника"""
    sample = 'train'
    suffix = '_flip'
    yolo_dataset = Path(cfg['dataset']['yolo_dataset'])
    source = Path(yolo_dataset, sample)
    flip_presence = check_flip_presence(source, suffix)
    if not flip_presence:
        formatter = YoloFormatter(cfg['classes']['class2pos'],
                        source, yolo_dataset)
        labels = get_lr_labels(source)
        formatter.save_data(make_lr_flip, get_bboxes, labels.image,
                            labels, sample, suffix=suffix)


def save_annotations(bboxes: np.ndarray, dst: Path):
    with open(dst, 'w') as f:
        for box in bboxes:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*box))


def add_suffix(name: Path, suffix: str) -> Path:
    new_name = str(name.with_suffix('')) + suffix + '.png'
    return Path(new_name)


def get_instance_names(image_name: Path, suffix: str) -> Tuple[Path, Path]:
    # filepath -> filename at Yolo dataset
    image_name = Path(str(image_name).replace('/', '_'))
    if suffix:
        image_name = add_suffix(image_name, suffix)
    else:
        image_name = image_name.with_suffix('.png')
    # image.* -> image.txt
    ann_name = Path(image_name).with_suffix('.txt')
    return image_name, ann_name


def get_bboxes(labels: pd.DataFrame, image: Path) -> np.ndarray:
    img_labels = labels[labels['image'] == image].values
    # Drop image_name column
    return img_labels[:, 1:]


def make_dataset_struct(yolo_dir: Path):
    for sample in ('train', 'valid'):
        Path.mkdir(
            Path(yolo_dir, sample), parents=True, exist_ok=True)


def train_val_split(labels: pd.DataFrame, test_size: float,
                    random_state: int) -> Dict[str, list]:
    """Разбиваем трейн на трейн и валидацию"""
    images = labels['image'].unique()
    train, test = train_test_split(
        images, test_size=test_size, random_state=random_state)
    return {'train': train, 'valid': test}


def get_yolo_info(image_dst: Path, class2pos: list) -> Dict[str, str]:
    data = {}
    data['train'] = f'train: {Path(image_dst, "train")}'
    data['val'] = f'val: {Path(image_dst, "valid")}'
    data['nc'] = f'nc: {len(class2pos)}'
    data['names'] = f'names: {class2pos}'
    return data


def get_class_lr_flip(clazz: int) -> int:
    """Отображение классов при LR flip"""
    if clazz == 3:
        return 2
    if clazz == 2:
        return 3
    return clazz


def make_lr_flip(source: Path, destination: Path) -> None:
    """LR flip и сохранение изображения"""
    image = read_image(source)
    image_lr_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    image_lr_flip.save(destination.with_suffix('.png'), 'PNG')


def get_lr_flipped_bboxes(labels: pd.DataFrame, image: Path) -> np.ndarray:
    """LR flip аннотаций"""
    img_labels = labels[labels['image'] == image].copy()
    img_labels['x_n'] = 1.0 - img_labels['x_n']
    img_labels['clazz'] = img_labels['clazz'].apply(
        lambda x: get_class_lr_flip(x))
    # Drop image_name column
    return img_labels.values[:, 1:]


def coords_normalization(labels: pd.DataFrame) -> pd.DataFrame:
    labels['box_width'] = labels['x_max'] - labels['x_min']
    labels['box_height'] = labels['y_max'] - labels['y_min']
    labels['x'] = ((labels['x_max'] + labels['x_min']) / 2).astype(int)
    labels['y'] = ((labels['y_max'] + labels['y_min']) / 2).astype(int)
    labels['x_n'] = labels['x'] / labels['width']
    labels['y_n'] = labels['y'] / labels['height']
    labels['w_n'] = labels['box_width'] / labels['width']
    labels['h_n'] = labels['box_height'] / labels['height']
    return labels[YOLO_COLUMNS]
