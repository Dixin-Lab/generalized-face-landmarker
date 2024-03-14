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
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default="/home/jiayi/Work_landmark/generalized_face_landmarker/snapshots/final_state.pt")

    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    parser.add_argument('--batch_size', help='how many samples in one batch', type=int, default=32)
    parser.add_argument(
        "--src_data",
        type=str,
        default='/home/jiayi/Work_landmark/Work_landmark/Data/300W',
        help="the path to the source dataset",
    )
    parser.add_argument(
        "--tgt_data",
        type=str,
        default='/home/jiayi/Work_landmark/Work_landmark/Data/CariFace_dataset',
        help="the path to the target dataset",
    )
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


def main_function_test():

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

    _, valid_loader = get_W300_dataloader(args.batch_size, args)

    checkpoint = torch.load(args.checkpoint)
    model.module.load_state_dict(checkpoint['warpA2B'])
    model.eval()

    src_err, tgt_err = [], []

    print(" [*] Testing on source domain!")
    with torch.no_grad():
        for i, meta in enumerate(valid_loader):
            input = meta['image']
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error*100.0)

            print(msg)
            src_err.append(error)
        print(" [*] finished on source domain!")

        # test on target domain
        print(" [*] Testing on target domain!")
        _, valid_loader = get_cartoon_dataloader(args.batch_size, args)

        for i, meta in enumerate(valid_loader):
            input = meta['image']
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error * 100.0)

            print(msg)
            tgt_err.append(error)
        print(" [*] finished on target domain!")

        return src_err, tgt_err


if __name__ == '__main__':
    src_err, tgt_err = main_function_test()

    cal = FR_AUC('300W')
    print("Mean Error on Source Domain: {:.3f}".format((np.mean(np.array(src_err)) * 100.0)))
    print("Failure Rate on Source Domain: {:.4f}, AUC: {:.4f}".format(cal.test(src_err, 0.08)[0], cal.test(src_err, 0.08)[1]))

    cal = FR_AUC('Cartoon')
    print("Mean Error on Target Domain: {:.3f}".format((np.mean(np.array(tgt_err)) * 100.0)))
    print("Failure Rate on Target Domain: {:.4f}, AUC: {:.4f}".format(cal.test(tgt_err, 0.08)[0], cal.test(tgt_err, 0.08)[1]))
    print('Done!')
