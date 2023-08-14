import argparse

from src.preprocessing.parser import AnnotationsParser
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.dataset import Dataset
from src.utils.utils import read_config
from src.utils.logging import create_logger


def main():
    logger = create_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/config/config.yaml')
    args, _ = parser.parse_known_args()

    logger.info(f"read config: {args.config}")
    cfg = read_config(args.config)

    # parse all XML to csv
    annotations_full = AnnotationsParser(source=cfg['source']).parse()

    ds = DataSplitter(cfg)

    # Split input
    train, test = ds.split(annotations_full)

    # Postpone test images
    ds.save_test_images(test)

    # Make Yolov5 format dataset
    Dataset(cfg).make(train)


if __name__ == '__main__':
    main()
