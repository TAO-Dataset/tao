# Modified from detectron2/demo/demo.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from pycocotools import mask
from script_utils.common import common_setup
from tqdm import tqdm

from tao.utils.parallel.fixed_gpu_pool import FixedGpuPool


def init_model(init_args, context):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(context['gpu'])
    context['predictor'] = DefaultPredictor(init_args['config'])


def infer(kwargs, context):
    predictor = context['predictor']
    image_path = kwargs['image_path']
    output_path = kwargs['output_path']
    img = read_image(str(image_path), format="BGR")

    predictions = predictor(img)
    predictions = predictions["instances"].get_fields()
    boxes_decoded = predictions["pred_boxes"].tensor.cpu().numpy().tolist()
    scores_decoded = predictions["scores"].cpu().numpy().tolist()
    classes_decoded = predictions["pred_classes"].cpu().numpy().tolist()
    masks_decoded = None
    if args.save_masks:
        masks_decoded = predictions["pred_masks"].cpu().numpy().astype(np.bool)
    save(boxes_decoded, scores_decoded, classes_decoded, masks_decoded,
         output_path)


def save(boxes_decoded, scores_decoded, classes_decoded, masks_decoded,
         results_path):
    predictions_decoded = {}
    predictions_decoded["instances"] = {
        "pred_boxes": boxes_decoded,
        "scores": scores_decoded,
        "pred_classes": classes_decoded,
    }
    if masks_decoded is not None:
        rles = mask.encode(
            np.array(masks_decoded.transpose((1, 2, 0)),
                     order='F',
                     dtype=np.uint8))
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        predictions_decoded['instances']['pred_masks'] = rles
    with open(results_path, 'wb') as f:
        pickle.dump(predictions_decoded, f)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if not args.save_masks:
        cfg.MODEL.MASK_ON = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output",
                        required=True,
                        type=Path,
                        help="Directory to save output pickles.")
    parser.add_argument("--config-file",
                        required=True,
                        type=Path,
                        help="path to config file")
    parser.add_argument('--gpus', default=[0], nargs='+', type=int)
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--save-masks', default=False, action='store_true')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    Path(args.output).mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output, args)
    # Prevent detectron from flooding terminal with messages.
    logging.getLogger('detectron2.checkpoint.c2_model_loading').setLevel(
        logging.WARNING)
    logging.getLogger('fvcore.common.checkpoint').setLevel(
        logging.WARNING)
    logger = logging.root

    cfg = setup_cfg(args)

    threads_per_worker = 4
    torch.set_num_threads(threads_per_worker)
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)

    all_files = args.root.rglob('*.jpg')

    # Arguments to init_model()
    init_args = {'config': cfg}

    # Tasks to pass to infer()
    infer_tasks = []
    i = 0
    for path in tqdm(all_files,
                     mininterval=1,
                     dynamic_ncols=True,
                     desc='Collecting frames'):
        relative = path.relative_to(args.root)
        output_pkl = (args.output / relative).with_suffix('.pkl')
        if output_pkl.exists():
            continue
        if i > 10:
            break
        i += 1
        output_pkl.parent.mkdir(exist_ok=True, parents=True)
        infer_tasks.append({'image_path': path, 'output_path': output_pkl})

    if len(args.gpus) == 1:
        context = {'gpu': args.gpus[0]}
        init_model(init_args, context)
        for task in tqdm(infer_tasks,
                         mininterval=1,
                         desc='Running detector',
                         dynamic_ncols=True):
            infer(task, context)
    else:
        pool = FixedGpuPool(
            args.gpus, initializer=init_model, initargs=init_args)
        list(
            tqdm(pool.imap_unordered(infer, infer_tasks),
                 total=len(infer_tasks),
                 mininterval=10,
                 desc='Running detector',
                 dynamic_ncols=True))
