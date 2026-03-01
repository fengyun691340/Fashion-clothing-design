# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py


import json
import os
import pathlib
import random
import sys
from typing import Tuple
from typing import List
PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes


class VitonHDDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            radius=5,
            caption_folder='logotrain.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            aug_type: str = "Resize",
            bbox_crop=True, 
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.aug_type = aug_type
        self.radius = radius
        self.sketch_threshold_range = sketch_threshold_range
        self.bbox_crop = bbox_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.custom_instance_prompts = False

        im_names = []
        c_names = []
        dataroot_names = []

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs3.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.toTensor = transforms.ToTensor()
    # def setup_transform(self):
    #     if self.bbox_crop:
    #         if self.aug_type == "Resize":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose = ([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #             ])
                
    #         elif self.aug_type == "Padding":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #             ])
    #         else:
    #             raise NotImplementedError("Do not support this augmentation")
        
    #     else:
    #         pixel_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ])
    #         guid_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #         ])
        
    #     return pixel_transform, guid_transform
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            # 注意：这里 self.image_size 未定义！应使用 (self.height, self.width)
            image_size = (self.height, self.width)
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # 文本提示
        captions = self.captions_dict[c_name.split('.')[0]]
        original_captions = captions

        # === 主图像 (instance image) ===
        image = Image.open(os.path.join(dataroot, self.phase, 'logo_image', im_name)).convert("RGB")
        image = image.resize((self.width, self.height))
        state = torch.get_rng_state()
        tgt_img = self.augmentation(image, self.pixel_transform, state)  # (3, H, W)

        # === 条件图像列表 (全部作为 cond) ===
        cond_imgs = []

        # 1. 草图
        if self.order == 'unpaired':
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                    os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png"))
        else:  # paired
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png"))
        im_sketch = Image.open(sketch_path).convert("RGB")
        im_sketch = im_sketch.resize((self.width, self.height))
        im_sketch = self.augmentation(im_sketch, self.pixel_transform, state)
        # cond_imgs.append(im_sketch)

        # # 2. OpenPose
        # pose_name = im_name.replace('.jpg', '_rendered.png')
        # im_pose = Image.open(os.path.join(dataroot, self.phase, 'openpose_img', pose_name)).convert("RGB")
        # im_pose = im_pose.resize((self.width, self.height))
        # im_pose = self.augmentation(im_pose, self.pixel_transform, state)
        # cond_imgs.append(im_pose)

        # 3. DensePose
        im_densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name)).convert("RGB")
        im_densepose = im_densepose.resize((self.width, self.height))
        im_densepose = self.augmentation(im_densepose, self.pixel_transform, state)
        # cond_imgs.append(im_densepose)

        # # 4. Texture
        # texture_name = im_name.replace('.jpg', '.png')
        # im_texture = Image.open(os.path.join(dataroot, self.phase, 'texture', texture_name)).convert("RGB")
        # im_texture = im_texture.resize((self.width, self.height))
        # im_texture = self.augmentation(im_texture, self.pixel_transform, state)
        # cond_imgs.append(im_texture)

        # 5. Logo
        logo = Image.open(os.path.join(dataroot, self.phase, 'logo_all', im_name)).convert("RGB")
        logo = logo.resize((self.width, self.height))
        logo = self.augmentation(logo, self.pixel_transform, state)
        

        # 6. Agnostic mask (as RGB image)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', im_name)).convert("RGB")
        im_mask = im_mask.resize((self.width, self.height))
        ref_img_vae = self.augmentation(im_mask, self.pixel_transform, state)
        cond_imgs.append(ref_img_vae)
        cond_imgs.append(logo)
        cond_imgs.append(im_sketch)
        cond_imgs.append(im_densepose)



        # === 拼接所有条件图：沿通道维度 concat ===
        # 每张图是 (3, H, W)，N 张 → (3*N, H, W)
        #cond_images = torch.cat(cond_imgs, dim=0)  # shape: (18, H, W) for 6 conditions

        # === 构建 DreamBooth 兼容的输出 ===
        example = {
            "pixel_values": tgt_img,           # (3, H, W) —— 目标生成图
            "cond_pixel_values": cond_imgs,           # (18, H, W) —— 所有条件图拼接
            "prompts": original_captions, # str or list of str
            # 如果你后续用 BucketBatchSampler，可加 bucket_idx；否则可省略
            "bucket_idx": 0,                      # 固定分辨率，设为 0
            "c_name":c_name,
            "im_name":im_name,
        }

        return example

    def __len__(self):
        return len(self.c_names)
# dataset_path="/home/sd/Harddisk/zj/control/ControlNet/HR-VITON"
# dataset = VitonHDDataset(
#             dataroot_path=dataset_path,
#             phase='train'
#         )
# print(dataset[0])
# print(len(dataset))
# for i in range(len(dataset)):

#     dataset[i]['pixel_values']
#     dataset[i]["cond_pixel_values"]
#     dataset[i]["prompts"]
# print(len(dataset))
class VitonHDDatasettest(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            radius=5,
            caption_folder='vitonhd.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            aug_type: str = "Resize",
            bbox_crop=True, 
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDatasettest, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.aug_type = aug_type
        self.radius = radius
        self.sketch_threshold_range = sketch_threshold_range
        self.bbox_crop = bbox_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.custom_instance_prompts = False

        im_names = []
        c_names = []
        dataroot_names = []

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairslogo.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs3.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.toTensor = transforms.ToTensor()
    # def setup_transform(self):
    #     if self.bbox_crop:
    #         if self.aug_type == "Resize":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose = ([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #             ])
                
    #         elif self.aug_type == "Padding":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #             ])
    #         else:
    #             raise NotImplementedError("Do not support this augmentation")
        
    #     else:
    #         pixel_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ])
    #         guid_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #         ])
        
    #     return pixel_transform, guid_transform
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            # 注意：这里 self.image_size 未定义！应使用 (self.height, self.width)
            image_size = (self.height, self.width)
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # 文本提示
        captions = self.captions_dict[c_name.split('_')[0]]
        original_captions = captions

        # === 主图像 (instance image) ===
        image = Image.open(os.path.join(dataroot, self.phase, 'image', im_name)).convert("RGB")
        image = image.resize((self.width, self.height))
        tgt_img=image
        # state = torch.get_rng_state()
        # tgt_img = self.augmentation(image, self.pixel_transform, state)  # (3, H, W)

        # === 条件图像列表 (全部作为 cond) ===
        cond_imgs = []

        # 1. 草图
        if self.order == 'unpaired':
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                    os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png"))
        else:  # paired
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png"))
        im_sketch = Image.open(sketch_path).convert("RGB")
        im_sketch = im_sketch.resize((self.width, self.height))
        #im_sketch = self.augmentation(im_sketch, self.pixel_transform, state)
        cond_imgs.append(im_sketch)

        # 2. OpenPose
        pose_name = im_name.replace('.jpg', '_rendered.png')
        im_pose = Image.open(os.path.join(dataroot, self.phase, 'openpose_img', pose_name)).convert("RGB")
        im_pose = im_pose.resize((self.width, self.height))
        # im_pose = self.augmentation(im_pose, self.pixel_transform, state)
        cond_imgs.append(im_pose)

        # 3. DensePose
        im_densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name)).convert("RGB")
        im_densepose = im_densepose.resize((self.width, self.height))
        # im_densepose = self.augmentation(im_densepose, self.pixel_transform, state)
        cond_imgs.append(im_densepose)

        # 4. Texture
        texture_name = im_name.replace('.jpg', '.png')
        im_texture = Image.open(os.path.join(dataroot, self.phase, 'texture', texture_name)).convert("RGB")
        im_texture = im_texture.resize((self.width, self.height))
        # im_texture = self.augmentation(im_texture, self.pixel_transform, state)
        cond_imgs.append(im_texture)

        # 5. Logo
        logo = Image.open(os.path.join(dataroot, self.phase, 'logo_yuan', im_name)).convert("RGB")
        logo = logo.resize((self.width, self.height))
        # logo = self.augmentation(logo, self.pixel_transform, state)
        cond_imgs.append(logo)

        # 6. Agnostic mask (as RGB image)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', im_name)).convert("RGB")
        im_mask = im_mask.resize((self.width, self.height))
        # ref_img_vae = self.augmentation(im_mask, self.pixel_transform, state)
        cond_imgs.append(im_mask)

        # === 拼接所有条件图：沿通道维度 concat ===
        # 每张图是 (3, H, W)，N 张 → (3*N, H, W)
        #cond_images = torch.cat(cond_imgs, dim=0)  # shape: (18, H, W) for 6 conditions

        # === 构建 DreamBooth 兼容的输出 ===
        example = {
            "pixel_values": tgt_img,           # (3, H, W) —— 目标生成图
            "cond_pixel_values": cond_imgs,           # (18, H, W) —— 所有条件图拼接
            "prompts": original_captions, # str or list of str
            # 如果你后续用 BucketBatchSampler，可加 bucket_idx；否则可省略
            "bucket_idx": 0,                      # 固定分辨率，设为 0
            "c_name":c_name,
            "im_name":im_name,
        }

        return example

    def __len__(self):
        return len(self.c_names)
    
class VitonHDDatasettestimage(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            radius=5,
            caption_folder='logotest.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            aug_type: str = "Resize",
            bbox_crop=True, 
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDatasettestimage, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.aug_type = aug_type
        self.radius = radius
        self.sketch_threshold_range = sketch_threshold_range
        self.bbox_crop = bbox_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.custom_instance_prompts = False

        im_names = []
        c_names = []
        dataroot_names = []

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.toTensor = transforms.ToTensor()
    # def setup_transform(self):
    #     if self.bbox_crop:
    #         if self.aug_type == "Resize":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose = ([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #             ])
                
    #         elif self.aug_type == "Padding":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #             ])
    #         else:
    #             raise NotImplementedError("Do not support this augmentation")
        
    #     else:
    #         pixel_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ])
    #         guid_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #         ])
        
    #     return pixel_transform, guid_transform
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            # 注意：这里 self.image_size 未定义！应使用 (self.height, self.width)
            image_size = (self.height, self.width)
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # 文本提示
        captions = self.captions_dict[c_name.split('_')[0]]
        original_captions = captions

        # === 主图像 (instance image) ===
        image = Image.open(os.path.join(dataroot, self.phase, 'logo_image', im_name)).convert("RGB")
        image = image.resize((self.width, self.height))
        # state = torch.get_rng_state()
        # tgt_img = self.augmentation(image, self.pixel_transform, state)  # (3, H, W)

        # === 条件图像列表 (全部作为 cond) ===
        cond_imgs = []

        # 1. 草图
        if self.order == 'unpaired':
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                    os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png"))
        else:  # paired
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png"))
        im_sketch = Image.open(sketch_path).convert("RGB")
        im_sketch = im_sketch.resize((self.width, self.height))
        # im_sketch = self.augmentation(im_sketch, self.pixel_transform, state)
        # cond_imgs.append(im_sketch)

        # # 2. OpenPose
        # pose_name = im_name.replace('.jpg', '_rendered.png')
        # im_pose = Image.open(os.path.join(dataroot, self.phase, 'openpose_img', pose_name)).convert("RGB")
        # im_pose = im_pose.resize((self.width, self.height))
        # im_pose = self.augmentation(im_pose, self.pixel_transform, state)
        # cond_imgs.append(im_pose)

        # 3. DensePose
        im_densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name)).convert("RGB")
        im_densepose = im_densepose.resize((self.width, self.height))
        # im_densepose = self.augmentation(im_densepose, self.pixel_transform, state)
        # cond_imgs.append(im_densepose)

        # # 4. Texture
        # texture_name = im_name.replace('.jpg', '.png')
        # im_texture = Image.open(os.path.join(dataroot, self.phase, 'texture', texture_name)).convert("RGB")
        # im_texture = im_texture.resize((self.width, self.height))
        # im_texture = self.augmentation(im_texture, self.pixel_transform, state)
        # cond_imgs.append(im_texture)

        # 5. Logo
        logo = Image.open(os.path.join(dataroot, self.phase, 'logo_all', im_name)).convert("RGB")
        logo = logo.resize((self.width, self.height))
        # logo = self.augmentation(logo, self.pixel_transform, state)
        

        # 6. Agnostic mask (as RGB image)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', im_name)).convert("RGB")
        im_mask = im_mask.resize((self.width, self.height))
        # ref_img_vae = self.augmentation(im_mask, self.pixel_transform, state)
        cond_imgs.append(im_mask)
        cond_imgs.append(logo)
        cond_imgs.append(im_sketch)
        cond_imgs.append(im_densepose)



        # === 拼接所有条件图：沿通道维度 concat ===
        # 每张图是 (3, H, W)，N 张 → (3*N, H, W)
        #cond_images = torch.cat(cond_imgs, dim=0)  # shape: (18, H, W) for 6 conditions

        # === 构建 DreamBooth 兼容的输出 ===
        example = {
            "pixel_values": image,           # (3, H, W) —— 目标生成图
            "cond_pixel_values": cond_imgs,           # (18, H, W) —— 所有条件图拼接
            "prompts": original_captions, # str or list of str
            # 如果你后续用 BucketBatchSampler，可加 bucket_idx；否则可省略
            "bucket_idx": 0,                      # 固定分辨率，设为 0
            "c_name":c_name,
            "im_name":im_name,
            
            
        }

        return example

    def __len__(self):
        return len(self.c_names)
    
class VitonHDDatasetreal(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            radius=5,
            caption_folder='logotrain.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            aug_type: str = "Resize",
            bbox_crop=True, 
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDatasetreal, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.aug_type = aug_type
        self.radius = radius
        self.sketch_threshold_range = sketch_threshold_range
        self.bbox_crop = bbox_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.custom_instance_prompts = False

        im_names = []
        c_names = []
        dataroot_names = []

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairslogo.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs3.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.toTensor = transforms.ToTensor()
    # def setup_transform(self):
    #     if self.bbox_crop:
    #         if self.aug_type == "Resize":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose = ([
    #                 transforms.Resize((self.height, self.width)),
    #                 transforms.ToTensor(),
    #             ])
                
    #         elif self.aug_type == "Padding":
    #             pixel_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5], [0.5]),
    #             ])
    #             guid_transform = transforms.Compose([
    #                 transforms.Lambda(self.resize_long_edge),
    #                 transforms.Lambda(self.padding_short_edge),
    #                 transforms.ToTensor(),
    #             ])
    #         else:
    #             raise NotImplementedError("Do not support this augmentation")
        
    #     else:
    #         pixel_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ])
    #         guid_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    #             transforms.ToTensor(),
    #         ])
        
    #     return pixel_transform, guid_transform
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            # 注意：这里 self.image_size 未定义！应使用 (self.height, self.width)
            image_size = (self.height, self.width)
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # 文本提示
        captions = self.captions_dict[c_name.split('.')[0]]
        original_captions = captions

        # === 主图像 (instance image) ===
        image = Image.open(os.path.join(dataroot, self.phase, 'logo_image', im_name)).convert("RGB")
        image = image.resize((self.width, self.height))
        state = torch.get_rng_state()
        tgt_img = self.augmentation(image, self.pixel_transform, state)  # (3, H, W)

        # === 条件图像列表 (全部作为 cond) ===
        cond_imgs = []

        # 1. 草图
        if self.order == 'unpaired':
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                    os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png"))
        else:  # paired
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png"))
        im_sketch = Image.open(sketch_path).convert("RGB")
        im_sketch = im_sketch.resize((self.width, self.height))
        im_sketch = self.augmentation(im_sketch, self.pixel_transform, state)
        # cond_imgs.append(im_sketch)

        # 3. DensePose
        im_densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name)).convert("RGB")
        im_densepose = im_densepose.resize((self.width, self.height))
        im_densepose = self.augmentation(im_densepose, self.pixel_transform, state)
        # cond_imgs.append(im_densepose)

        # 5. Logo
        logo = Image.open(os.path.join(dataroot, self.phase, 'logo_all', im_name)).convert("RGB")
        logo = logo.resize((self.width, self.height))
        logo = self.augmentation(logo, self.pixel_transform, state)
        

        # 6. Agnostic mask (as RGB image)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', im_name)).convert("RGB")
        im_mask = im_mask.resize((self.width, self.height))
        ref_img_vae = self.augmentation(im_mask, self.pixel_transform, state)
        cond_imgs.append(ref_img_vae)
        cond_imgs.append(logo)
        cond_imgs.append(im_sketch)
        cond_imgs.append(im_densepose)



        # === 拼接所有条件图：沿通道维度 concat ===
        # 每张图是 (3, H, W)，N 张 → (3*N, H, W)
        #cond_images = torch.cat(cond_imgs, dim=0)  # shape: (18, H, W) for 6 conditions

        # === 构建 DreamBooth 兼容的输出 ===
        example = {
            "pixel_values": tgt_img,           # (3, H, W) —— 目标生成图
            "cond_pixel_values": cond_imgs,           # (18, H, W) —— 所有条件图拼接
            "prompts": original_captions, # str or list of str
            # 如果你后续用 BucketBatchSampler，可加 bucket_idx；否则可省略
            "bucket_idx": 0,                      # 固定分辨率，设为 0
        }

        return example

    def __len__(self):
        return len(self.c_names)
# dataset_path="/home/sd/Harddisk/zj/control/ControlNet/HR-VITON"
# dataset = VitonHDDatasettestimage(
#             dataroot_path=dataset_path,
#             phase='test'
#         )
# print(dataset[0]["prompts"])
# print(dataset[0]["prompts"]+' '+'image is a bad image')
# print(len(dataset))
# for i in range(len(dataset)):

#     dataset[i]['pixel_values']
#     dataset[i]["cond_pixel_values"]
#     dataset[i]["prompts"]
# # print(len(dataset))
class VitonHDDatasetdpo(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            radius=5,
            caption_folder='vitonhd.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            aug_type: str = "Resize",
            bbox_crop=True, 
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDatasetdpo, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.aug_type = aug_type
        self.radius = radius
        self.sketch_threshold_range = sketch_threshold_range
        self.bbox_crop = bbox_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.custom_instance_prompts = False

        im_names = []
        c_names = []
        dataroot_names = []

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs3.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.toTensor = transforms.ToTensor()
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            # 注意：这里 self.image_size 未定义！应使用 (self.height, self.width)
            image_size = (self.height, self.width)
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # 文本提示
        captions = self.captions_dict[c_name.split('_')[0]]
        original_captions = captions

        # === 主图像 (instance image) ===
        image = Image.open(os.path.join(dataroot, self.phase, 'logo_image', im_name)).convert("RGB")
        image = image.resize((self.width, self.height))
        state = torch.get_rng_state()
        tgt_img = self.augmentation(image, self.pixel_transform, state)  # (3, H, W)

        # === 条件图像列表 (全部作为 cond) ===
        cond_imgs = []

        # 1. 草图
        if self.order == 'unpaired':
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                    os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png"))
        else:  # paired
            sketch_path = os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png"))
        im_sketch = Image.open(sketch_path).convert("RGB")
        im_sketch = im_sketch.resize((self.width, self.height))
        im_sketch = self.augmentation(im_sketch, self.pixel_transform, state)
        # cond_imgs.append(im_sketch)

        # # 2. OpenPose
        # pose_name = im_name.replace('.jpg', '_rendered.png')
        # im_pose = Image.open(os.path.join(dataroot, self.phase, 'openpose_img', pose_name)).convert("RGB")
        # im_pose = im_pose.resize((self.width, self.height))
        # im_pose = self.augmentation(im_pose, self.pixel_transform, state)
        # cond_imgs.append(im_pose)

        # 3. DensePose
        im_densepose = Image.open(os.path.join(dataroot, self.phase, 'image-densepose', im_name)).convert("RGB")
        im_densepose = im_densepose.resize((self.width, self.height))
        im_densepose = self.augmentation(im_densepose, self.pixel_transform, state)
        # cond_imgs.append(im_densepose)

        # # 4. Texture
        # texture_name = im_name.replace('.jpg', '.png')
        # im_texture = Image.open(os.path.join(dataroot, self.phase, 'texture', texture_name)).convert("RGB")
        # im_texture = im_texture.resize((self.width, self.height))
        # im_texture = self.augmentation(im_texture, self.pixel_transform, state)
        # cond_imgs.append(im_texture)

        # 5. Logo
        logo = Image.open(os.path.join(dataroot, self.phase, 'logo_all', im_name)).convert("RGB")
        logo = logo.resize((self.width, self.height))
        logo = self.augmentation(logo, self.pixel_transform, state)
        # bad case 
        badimagename=im_name+"_"+im_name+'.png'
        badimage = Image.open(os.path.join("/home/sd/Harddisk/zj/flux/flux_lora/vitonhd/prompt64-7000/37000/train", badimagename)).convert("RGB")
        badimage = badimage.resize((self.width, self.height))
        badimage = self.augmentation(badimage, self.pixel_transform, state)

        

        # 6. Agnostic mask (as RGB image)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'agnostic-v3.2', im_name)).convert("RGB")
        im_mask = im_mask.resize((self.width, self.height))
        ref_img_vae = self.augmentation(im_mask, self.pixel_transform, state)
        cond_imgs.append(ref_img_vae)
        cond_imgs.append(logo)
        cond_imgs.append(im_sketch)
        cond_imgs.append(im_densepose)



        # === 拼接所有条件图：沿通道维度 concat ===
        # 每张图是 (3, H, W)，N 张 → (3*N, H, W)
        #cond_images = torch.cat(cond_imgs, dim=0)  # shape: (18, H, W) for 6 conditions

        # === 构建 DreamBooth 兼容的输出 ===
        example = {
            "pixel_values": tgt_img,           # (3, H, W) —— 目标生成图
            "bad_pixel_values": badimage,           # (3, H, W) —— 目标生成图
            "cond_pixel_values": cond_imgs,           # (18, H, W) —— 所有条件图拼接
            "prompts": original_captions, # str or list of str
            # 如果你后续用 BucketBatchSampler，可加 bucket_idx；否则可省略
            "bucket_idx": 0,                      # 固定分辨率，设为 0
            "c_name":c_name,
            "im_name":im_name,
        }

        return example

    def __len__(self):
        return len(self.c_names)
# dataset_path="/home/sd/Harddisk/zj/control/ControlNet/HR-VITON"
# dataset = VitonHDDatasetdpo(
#             dataroot_path=dataset_path,
#             phase='train'
#         )
# print(dataset[0])