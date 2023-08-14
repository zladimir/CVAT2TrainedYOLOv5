import argparse
from src.utils.utils import read_config
from src.train.model_trainer import ModelTrainer
from src.preprocessing.dataset import flip_yolo_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/config/config.yaml')
    parser.add_argument('--yolo_engine', type=str, default='/root/yolov5')
    args, _ = parser.parse_known_args()

    cfg = read_config(args.config)

    # Flip YOLO dataset from external source
    if cfg['model']['lr_flip']:
        flip_yolo_dataset(cfg)

    trainer = ModelTrainer(cfg, args.yolo_engine)
    trainer.train()


if __name__ == '__main__':
    main()
