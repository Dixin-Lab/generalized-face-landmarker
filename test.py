import argparse

from Landmark2.Config import cfg
from Landmark2.Config import update_config

from utils import create_logger
from Landmark2.model import Sparse_alignment_network

import torch
import pprint

from data.W300 import get_W300_dataloader
from data.Cartoon import get_cartoon_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Test Sparse Facial Network')

    # landmark_detector
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default="D:/PycharmProjects/HRnet_warp_in_turn/checkpoints_update_in_turn_perceptual/params_0000040.pt")

    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    parser.add_argument('--batch_size', help='how many samples in one batch', type=int, default=32)

    args = parser.parse_args()

    return args


def calcuate_loss(name, pred, gt, trans):

    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(gt[36, :] - gt[45, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[17, :] - gt[16, :])
    else:
        raise ValueError('Wrong Dataset')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real

import numpy as np
from scipy.integrate import simps

class FR_AUC:
    def __init__(self, thresh):
        self.thresh = thresh

    def __repr__(self):
        return "FR_AUC()"

    def test(self, nmes, thres=None, step=0.0001):
        if thres is None:
            thres = self.thresh

        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
        fr = 1.0 - ys[-1]
        auc = simps(ys, x=xs) / thres
        return [round(fr, 4), round(auc, 6)]


def main_function_source():

    args = parse_args()
    update_config(cfg, args)

    # create logger
    logger = create_logger(cfg)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = Sparse_alignment_network(cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
                                       cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                       cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                       cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg).to('cuda')

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    _, valid_loader = get_W300_dataloader(args.batch_size)

    checkpoint = torch.load(args.checkpoint)
    model.module.load_state_dict(checkpoint['warpA2B'])
    model.eval()

    error_list = []
    cal = FR_AUC('300W')

    with torch.no_grad():
        for i, (input, meta) in enumerate(valid_loader):

            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error*100.0)

            print(msg)
            error_list.append(error)

        print("finished")
        print("Mean Error: {:.3f}".format((np.mean(np.array(error_list)) * 100.0)))
        print("Failure Rate: {:.4f}, AUC: {:.4f}".format(cal.test(error_list, 0.08)[0], cal.test(error_list, 0.08)[1]))

def main_function_target():

    args = parse_args()
    update_config(cfg, args)

    # create logger
    logger = create_logger(cfg)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = Sparse_alignment_network(cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
                                       cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                       cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                       cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg).to('cuda')

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    _, valid_loader = get_cartoon_dataloader(args.batch_size)

    checkpoint = torch.load(args.checkpoint)
    model.module.load_state_dict(checkpoint['warpA2B'])
    model.eval()

    error_list = []
    cal = FR_AUC('CariFace')

    with torch.no_grad():
        for i, (input, meta) in enumerate(valid_loader):

            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error*100.0)

            print(msg)
            error_list.append(error)

        print("finished")
        print("Mean Error: {:.3f}".format((np.mean(np.array(error_list)) * 100.0)))
        print("Failure Rate: {:.4f}, AUC: {:.4f}".format(cal.test(error_list, 0.08)[0], cal.test(error_list, 0.08)[1]))

if __name__ == '__main__':
    print(" [*] Testing on source domain!")
    main_function_source()

    print(" [*] Testing on target domain!")
    main_function_target()
