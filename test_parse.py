from src.preprocessing.parser import AnnotationsParser
from src.utils.utils import save_csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images', type=str, default='/test/images')
    parser.add_argument('--test_labels', type=str, default='/test/test.csv')

    args, _ = parser.parse_known_args()

    # Parse all XML to csv and save to args.test_labels
    test_annotations = AnnotationsParser(
        source=args.test_images,
        relative_path=True
    ).parse()
    save_csv(test_annotations, args.test_labels)


if __name__ == '__main__':
    main()
