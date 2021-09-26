'''

'''
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input-file', '-i')

    args = parser.parse_args()

    img = Image.open(args.input_file)
    img = img.convert('RGB')

    img.save(args.input_file)


if __name__ == '__main__':
    main()
