import copy
import logging
import cv2
import numpy
import torch
import numpy as np
import os
import Landmark2.utils as utils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Landmark2.Config import cfg

logger = logging.getLogger(__name__)


class W300_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root
        self.number_landmarks = cfg.W300.NUM_POINT

        self.Fraction = cfg.W300.FRACTION

        self.Data_Format = cfg.W300.DATA_FORMAT

        self.Transform = transform

        # load frontal images and annotations
        if is_train:
            self.root_landmark = os.path.join(root, 'frontal_train_label')
            self.annotation_file = os.path.join(root, 'frontal_train_list.txt')
        else:
            self.root_landmark = root
            self.annotation_file = os.path.join(root, 'test_list.txt')

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            if self.is_train:
                temp_name = os.path.join(self.root, 'frontal_train', temp_info)
                Points = np.load(os.path.join(self.root_landmark, temp_info + ".npy"))
            else:
                temp_name = os.path.join(self.root, temp_info)
                Points = np.genfromtxt(os.path.join(self.root_landmark, temp_info[:-3] + "pts"), skip_header=3,
                                       skip_footer=1, delimiter=' ') - 1.0

            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': Points})

        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        Annotated_Points = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        if self.Transform is not None:
            input = self.Transform(input)
        meta = {
            'image': input,
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'points': Points / (self.Image_size),
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'angle': 0.0,
            'Translation': [0.0, 0.0],
        }
        return meta


def get_W300_dataloader(batch_size, opt):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )

    train_dataset = W300_Dataset(cfg, opt.src_data, True,
                                 transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataset = W300_Dataset(cfg, opt.src_data, False,
                                transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return train_loader, test_loader
