import argparse
import os.path

import torch.optim

from Landmark2.model import Sparse_alignment_network
from Landmark2.Config import cfg
import torch.nn as nn

from data.Combine_new import get_combine_dataloader # dataloader for source data(with labels) and target data(with pseudo labels)
from data.W300 import get_W300_dataloader # dataloader for source data (with labels)
from data.Cartoon import get_cartoon_dataloader # dataloader for target data (without labels)

from warp import *
from utils import *
import cv2
import kornia as K
import numpy as np
from Landmark2.test import main_function_test
from Landmark2.backbone import Alignment_Loss
from Landmark2.test import save_point


def build_model(opt):
    device = torch.device("cuda:{}".format(opt.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)
    warpA2B = Sparse_alignment_network(cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
                                       cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                       cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                       cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg).to(device)

    # load 300W_pretrain_model
    checkpoint_file = opt.pretrain_path
    checkpoint = torch.load(checkpoint_file)
    warpA2B.load_state_dict(checkpoint)

    for params in warpA2B.parameters():
        params.requires_grad = True

    W_optim = torch.optim.Adam(warpA2B.parameters(), lr=1e-4)
    L_optim = torch.optim.Adam(warpA2B.parameters(), lr=1e-3)
    return warpA2B, W_optim, L_optim


def get_gradient_loss(pre, target, MSE):
    pre = K.color.rgb_to_grayscale(pre)
    target = K.color.rgb_to_grayscale(target)
    pre_laplacian: torch.Tensor = K.filters.sobel(pre)
    target_laplacian: torch.Tensor = K.filters.sobel(target)
    return MSE(pre_laplacian, target_laplacian)

def get_loss(pre, target, loss):
    return loss(pre, target)


def train(warpA2B, W_optim, L_optim, opt):
    device = torch.device("cuda:{}".format(opt.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    trainA_loader, testA_loader = get_W300_dataloader(opt)
    trainB_loader, testB_loader = get_cartoon_dataloader(opt)
    trainB_generate_loader, testB_generate_loader = get_cartoon_dataloader(opt)

    MSE_loss = nn.MSELoss().to(device)
    landmark_loss = Alignment_Loss(cfg)

    for step in range(opt.epoch):
        warpA2B.train()
        if (step // opt.update_frequency) % 2 == 0:
            print('---------start training warper----------')
            for batch, (meta_A, meta_B) in enumerate(zip(trainA_loader, trainB_loader)):
                real_A, real_B = meta_A['image'].to(device), meta_B['image'].to(device)
                lm_A = meta_A['points'].to(device)

                W_optim.zero_grad()
                pre_B_lm = warpA2B(real_B)[2][:, -1, :, :]
                # prepare src and dst points for TPS warping
                points_src = pre_B_lm[:, :, [1, 0]] * 255.0
                points_dst = lm_A[:, :, [1, 0]] * 255.0
                warp_image, warp_filed = sparse_image_warp(real_A, points_dst, points_src)

                W_loss = get_gradient_loss(warp_image, real_B, MSE_loss)
                print("[%5d/%5d/%5d] W_loss: %.8f" % (batch, step, opt.epoch, W_loss))
                W_loss.backward()
                W_optim.step()

            if (step + 1) % opt.update_frequency == 0:
                save_path = os.path.join(opt.save_data_path, 'cariface_points_pesudo_' + str(step) + '/')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_point(warpA2B, trainB_generate_loader, save_path)
                save_point(warpA2B, testB_generate_loader, save_path)

        else:
            print('---------start training landmarker----------')
            # use mixed dataloader (source data + true label && target data + pseudo label) to update face landmarker
            train_combine_loader, test_combine_loader = get_combine_dataloader(save_path)
            for batch, meta in enumerate(train_combine_loader):
                real = meta['image'].to(device)
                lm = meta['points'].to(device)
                L_optim.zero_grad()
                pre = warpA2B(real)
                loss = landmark_loss(pre[0], lm) * 0.2 + landmark_loss(pre[1], lm) * 0.3 + landmark_loss(pre[2], lm) * 0.5
                loss.backward()
                L_optim.step()
                print("[%5d/%5d/%5d] L_loss: %.8f" % (batch, step, opt.epoch, loss))

        if step % 5 == 0:
            print('---------evaluating warpA2B and save checkpoints----------')
            B2A = np.zeros((256 * 3, 0, 3))
            test_sample_num = 5
            warpA2B.eval()
            for index in range(test_sample_num):
                try:
                    meta_A = next(testA_iter)
                except:
                    testA_iter = iter(testA_loader)
                    meta_A = next(testA_iter)

                try:
                    meta_B = next(testB_iter)
                except:
                    testB_iter = iter(testB_loader)
                    meta_B = next(testB_iter)
                real_A, real_B = meta_A['image'].to(device), meta_B['image'].to(device)
                lm_A, lm_B = meta_A['points'].to(device), meta_B['points'].to(device)
                pre_B_lm = warpA2B(real_B)[2][:, -1, :, :]

                points_src = pre_B_lm[:, :, [1, 0]] * 255.0
                points_dst = lm_A[:, :, [1, 0]] * 255.0
                warp_image, warp_filed = sparse_image_warp(real_A, points_dst, points_src)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))) * 255.0,
                                                           RGB2BGR(tensor2numpy(denorm(real_B[0]))) * 255.0,
                                                           draw_landmark(warp_image[0], pre_B_lm[0])), 0)), 1)
            if (step+1) % 15 == 0:
                cv2.imwrite(os.path.join(opt.result_dir, 'A2B_%5d_%1d.png')% (step, index), B2A)

            params = {}
            params['warpA2B'] = warpA2B.state_dict()
            torch.save(params, os.path.join(opt.snapshot_dir, 'params_%07d.pt' % step))

        params = {}
        params['warpA2B'] = warpA2B.state_dict()
        torch.save(params, os.path.join(opt.snapshot_dir, 'final_state.pt'))

def test_warp(warpA2B, loader):
    nme = main_function_test(warpA2B, loader)
    print(nme)
    return nme



"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(args.result_dir)
    check_folder(args.snapshot_dir)
    check_folder(args.save_data_path)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many samples for one batch",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="total number of epoch",
    )
    parser.add_argument(
        "--update_frequency",
        type=int,
        default=5,
        help="update warpA2B as warper and landmarker in turn for every n epochs",
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default='Landmark2/model_best.pth',
        help="the path to the source pretrained model weights",
    )
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
    parser.add_argument(
        "--save_data_dir",
        type=str,
        default='./pseudo_data',
        help="the path to save the pseudo labels",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='./result',
        help="the path to save the validation results",
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default='./snapshots',
        help="the path to save the checkpoint file",
    )

    opt = parser.parse_args()

    warpA2B, W_optim, L_optim = build_model(opt)
    train(warpA2B, W_optim, L_optim, opt)
    print(" [*] Training finished!")