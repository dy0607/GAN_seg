# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from datasets import BaseDataset
from datasets import IterDataLoader
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    seg_color = colorEncode(seg, colors)
    pred_color = colorEncode(pred, colors)
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    freq = torch.zeros((150)).cuda()
    total = 0

    for batch_data in loader:
        # process data

        batch_data = batch_data['image'].cuda()
        total += batch_data.shape[0]

        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (256, 256)
            
            pred = segmentation_module(batch_data, segSize=segSize)
            freq += pred.sum(dim=(0, 2, 3))

        time_meter.update(time.perf_counter() - tic)

        pbar.update(1)

    print(freq / total)

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, fixed=True)

    # Dataset and Loader
    
    resolution = 256
    data = dict(
        num_workers=4,
        repeat=1,
        train=dict(root_dir='data/ADEChallengeData2016/images/training', data_format='dir',
                    resolution=resolution, mirror=0.5),
        val=dict(root_dir='data/ADEChallengeData2016/images/validation', data_format='dir',
                    resolution=resolution),
    )

    mode = 'train'
    dataset = BaseDataset(**data[mode])
    train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
            num_workers=data.get('num_workers', 2))

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, train_loader, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("--imgs", required=True, type=str, help="an image path, or a directory name")
    parser.add_argument("--cfg", default="config_seg/ade20k-hrnetv2.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--gpu", default=2, type=int, help="gpu id for evaluation")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
