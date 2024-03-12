import torch.utils.data
from torchvision.transforms import transforms
from Landmark2.Config import cfg
import Landmark2.utils as utils

import copy
import logging
import cv2
import numpy
import torch
import numpy as np
import os

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Combine_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root
        self.number_landmarks = cfg.W300.NUM_POINT
        self.flip_index = np.genfromtxt(os.path.join("/home/jiayi/Work_landmark/Data/300W/", "Mirror.txt"), dtype=int,
                                        delimiter=',')

        self.Fraction = cfg.W300.FRACTION
        self.Translation_Factor = cfg.W300.TRANSLATION
        self.Rotation_Factor = cfg.W300.ROTATION
        self.Scale_Factor = cfg.W300.SCALE
        self.Occlusion_Mean = cfg.W300.OCCLUSION_MEAN
        self.Occlusion_Std = cfg.W300.OCCLUSION_STD
        self.Flip = cfg.W300.FLIP
        self.Occlusion = cfg.W300.OCCLUSION
        self.Transfer = cfg.W300.CHANNEL_TRANSFER

        self.Heatmap_size = cfg.MODEL.HEATMAP

        self.Data_Format = cfg.W300.DATA_FORMAT

        self.Transform = transform

        if is_train:
            self.annotation_file_300W = os.path.join("/home/jiayi/Work_landmark/Data/300W/", 'train_list.txt')
            self.annotation_file_Cartoon = os.path.join("/home/jiayi/Work_landmark/Data/AF_dataset/",
                                                        'train_list.txt')
        else:
            self.annotation_file_300W = os.path.join("/home/jiayi/Work_landmark/Data/300W/", 'test_list.txt')
            self.annotation_file_Cartoon = os.path.join("/home/jiayi/Work_landmark/Data/AF_dataset/",
                                                        'test_list.txt')
        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []
        if not os.path.exists(self.annotation_file_300W):
            raise FileNotFoundError(f"File not found: {self.annotation_file_300W}")

        with open(self.annotation_file_300W) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_name = os.path.join("/home/jiayi/Work_landmark/Data/300W/", temp_info)

            Points = np.genfromtxt(temp_name[:-3] + 'pts', skip_header=3, skip_footer=1, delimiter=' ') - 1.0

            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': Points})

        with open(self.annotation_file_Cartoon) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_name = os.path.join("/home/jiayi/Work_landmark/Data/AF_dataset/images", temp_info)
            Points = np.load(os.path.join(self.root, temp_info[:-3] + 'npy'))

            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': Points})
        return Data_base

    def Image_Flip(self, Img, GT):
        Mirror_GT = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, numpy.array(Mirror_GT)

    def Channel_Transfer(self, Img, Flag):
        if Flag == 1:
            Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        return Img

    def Create_Occlusion(self, Img):
        Occlusion_width = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_high = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_x = np.random.randint(0, self.Image_size - Occlusion_width)
        Occlusion_y = np.random.randint(0, self.Image_size - Occlusion_high)

        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 0] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 1] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 2] = \
            np.random.randint(0, 256)

        return Img

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

        if self.is_train == True:
            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor,
                            self.Fraction + Scale_Factor)

            Translation_X = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor,
                                    Translation_X_Factor)
            Translation_Y = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor,
                                    Translation_Y_Factor)

            trans = utils.get_transforms(BBox, Scale, angle, self.Image_size,
                                         shift_factor=[Translation_X, Translation_Y])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    input, Points = self.Image_Flip(input, Points)

            if self.Transfer is True:
                Transfer_Flag = np.random.randint(0, 5)
                input = self.Channel_Transfer(input, Transfer_Flag)

            if self.Occlusion is True:
                Occlusion_Flag = np.random.randint(0, 2)
                if Occlusion_Flag == 1:
                    input = self.Create_Occlusion(input)

            if self.Transform is not None:
                input = self.Transform(input)

            meta = {
                'image': input,
                'Img_path': Img_path,
                'points': Points / (self.Image_size),
                'BBox': BBox,
                'trans': trans,
                'Scale': Scale,
                'angle': angle,
                'Translation': [Translation_X, Translation_Y]}

            return meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(Points.shape[0]):
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


def get_combine_dataloader(root_landmark, batch_size):
    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    train_dataset = Combine_Dataset(cfg=cfg, root=root_landmark, is_train=True, transform=normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    cartoon_dataset = Combine_Dataset(cfg=cfg, root="/home/jiayi/Work_landmark/Data/AF_dataset/landmarks", is_train=False,
                                      transform=normalize)
    cartoon_loader = torch.utils.data.DataLoader(
        cartoon_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, cartoon_loader
