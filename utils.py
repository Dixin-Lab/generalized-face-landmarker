import logging
import time
from pathlib import Path

from scipy import misc
import os, cv2, torch
import numpy as np

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def draw_landmark(img, landmark):
    img = RGB2BGR(tensor2numpy(img * 0.5 + 0.5)) * 255.0
    pts = landmark.detach().cpu() * 255.0
    pts = np.array(pts, dtype=np.int32)
    for i in range(68):
        cv2.circle(img, (pts[i, 0], pts[i, 1]), 1, (0, 0, 255), 2)

    return img

def create_logger(cfg, cfg_name, result_dir, loopname, phase='train'):
    root_output_dir = Path(os.path.join(os.getcwd(), result_dir, loopname, cfg.OUTPUT_DIR))
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = '300W2Cartoon'
    model = cfg.MODEL.NAME
    # cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

from torch.nn import functional as F
def recon_loss(pred, true):
    return F.l1_loss(pred, true)

def TV(x):
    ell =  torch.pow(torch.abs(x[:,:,1:,: ] - x[:,:,0:-1,:  ]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,: ,1:] - x[:,:,:  ,0:-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,1:] - x[:,:, :-1, :-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,:-1] - x[:,:,:-1,1:]), 2).mean()
    ell /= 4.
    return ell