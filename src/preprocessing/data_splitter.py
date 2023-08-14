import pandas as pd
from typing import Tuple
from pathlib import Path
from src.utils.utils import save_csv, image_copy
from src.utils.logging import create_logger


class DataSplitter:
    def __init__(self, cfg):
        self.logger = create_logger()
        self.source = Path(cfg['source'])
        self.test_path = Path(cfg['test']['test_labels'])
        self.test_images = Path(cfg['test']['test_images'])
        self.test_query = cfg['train_test_split']['test_query']
        self.train_query = f'not ({self.test_query})'

    def split(self, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test"""
        labels['camera'] = labels['image'].apply(
            lambda x: x.relative_to(self.source).parts[0])
        labels['date'] = labels['image'].apply(
            lambda x: x.relative_to(self.source).parts[1])
        train = labels.query(self.train_query).copy()
        self.logger.info(
            f"train_query: {self.train_query}. rows: {train.shape[0]}")
        test = labels.query(self.test_query).copy()
        self.logger.info(
            f"test_query: {self.test_query}. rows: {test.shape[0]}")
        return train, test

    def save_test_images(self, test: pd.DataFrame):
        # Create relative to source path
        test['image'] = test['image'].apply(
            lambda x: x.relative_to(self.source))
        if not test.empty:
            save_csv(test, self.test_path)
        for img in test.image.unique():
            src = Path(self.source, img)
            dst = Path(self.test_images, img)
            image_copy(src, dst)
