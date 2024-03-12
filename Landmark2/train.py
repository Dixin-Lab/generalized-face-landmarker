import argparse

from Landmark2.Config import cfg
from Landmark2.Config import update_config

from Landmark2.utils import create_logger
from Landmark2.utils import save_checkpoint
from Landmark2.model import Sparse_alignment_network
from Landmark2.backbone import Alignment_Loss
from Landmark2.utils import get_optimizer
from Landmark2.tools import train

from tensorboardX import SummaryWriter

import torch
import pprint
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def initial_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

  
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    model = Sparse_alignment_network(cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
                                cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg)


    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    best_perf = 100.0
    # best_model = False
    last_epoch = -1
    loss_function_2 = Alignment_Loss(cfg).cuda()
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    print(checkpoint_file)
    checkpoint_file=""
    checkpoint_file="/home/haotian/work/SLPT_Training-main/Checkpoint/300W_anime_only/Sparase_alignment/final_state.pth"
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.module.load_state_dict(checkpoint)
        # print(checkpoint.keys())
        # model.load_state_dict(checkpoint['state_dict'])
        # print("hello")
        # optimizer.load_state_dict(checkpoint['optimizer'])


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    return model,optimizer,lr_scheduler,loss_function_2
def main_function(model,optimizer,lr_scheduler,loss_function_2,input, ground_truth):
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    landmarks,R_loss_1,R_loss_2,R_loss_3,loss=train(cfg,input, ground_truth, model, loss_function_2, optimizer,  writer_dict)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, model, final_output_dir)

    lr_scheduler.step()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    return landmarks,R_loss_1,R_loss_2,R_loss_3,loss


if __name__ == '__main__':
    initial_function()