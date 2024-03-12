import argparse

from Landmark2.Config import cfg
from Landmark2.Config import update_config

from Landmark2.utils import create_logger
from Landmark2.model import Sparse_alignment_network
from Landmark2.utils import AverageMeter


from tensorboardX import SummaryWriter

import torch
import cv2
import numpy as np
import pprint
import os

import torchvision.transforms as transforms

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

def calcuate_loss_ip(name, pred, gt, trans):

    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(np.mean(gt[36:42], axis=0) - np.mean(gt[42:48], axis=0))
    elif name == 'COFW':
        norm = np.linalg.norm(gt[17, :] - gt[16, :])
    else:
        raise ValueError('Wrong Dataset')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real


def main_function_test(model,valid_loader):


    error_list = AverageMeter()
    ip_list=AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i,  meta in enumerate(valid_loader):
            input=meta['image']
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]
            
            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()
            
            error = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            error_list.update(error, input.size(0))
            ip_loss=calcuate_loss_ip(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            ip_list.update(ip_loss,input.size(0))
            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error_list.avg * 100.0)

            print(msg)
            
        print("finished")
        print("Mean Error: {:.3f}".format(error_list.avg * 100.0))
    return error_list.avg*100.0, ip_list.avg*100


def save_point(model, valid_loader, save_path_points):
    error_list = AverageMeter()
    ip_list = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, meta in enumerate(valid_loader):
            input = meta['image']
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]
            image_path = meta['Img_path'][0].split("/")[-1]
            outputs_initial = model(input.cuda())

            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            error, pre = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            np.save(os.path.join(save_path_points, image_path + '.npy'), pre)

            error_list.update(error, input.size(0))
            ip_loss = calcuate_loss_ip(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            ip_list.update(ip_loss, input.size(0))
            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {error:.3f}%\t'.format(
                i, len(valid_loader), error=error_list.avg * 100.0)

            print(msg)

        print("pseudo landmark points saved!")
        print("Mean Error: {:.3f}".format(error_list.avg * 100.0))
    return error_list.avg * 100.0, ip_list.avg * 100
