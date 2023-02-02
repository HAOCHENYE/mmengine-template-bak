# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine_template.infer import CustomInferencer
from mmengine_template.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-file', default='result', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmdet.
    register_all_modules()

    inferencer = CustomInferencer(args.config, args.checkpoint, args.device)
    inferencer(args.img)


if __name__ == '__main__':
    args = parse_args()
    main(args)
