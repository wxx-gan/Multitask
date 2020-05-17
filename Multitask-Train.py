# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib,utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
iter_num=0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes 数据集类别数

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.修改为自己图片尺寸
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 576

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (6 * 6, 12 * 6, 24 * 6, 48 * 6, 96 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 30

config = ShapesConfig()
config.display()

class DrugDataset(utils.Dataset):#继承utils.dataset
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(),Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path): # 配置一些self图像基本信息
        # Add classes
        #add_class(source, class_id, class_name) source只是一个名字,整个网络并没有用到source,source不好改,改动起来太费事
        #self.add_class("shapes", 0, "BG")
        self.add_class("shapes", 1, "Orange") # 黑色素瘤

        self.add_sonclass("shapes",0,"BG")  #BG
        self.add_sonclass("shapes",1,"bad") #坏
        self.add_sonclass("shapes",2,"good") #好

        self.add_grandclass("shapes",0,"BG") #BG
        self.add_grandclass("shapes",1,"unripe") #未熟
        self.add_grandclass("shapes",2,"rare") #半熟
        self.add_grandclass("shapes",3,"ripe")  #熟

        for i in range(count):
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "/info.yaml"

            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "/img.png")
            try:
                print(dataset_root_path + "labelme_json/" + filestr + "/img.png" + '  ' + str(cv_img.shape))
            except:
                print('Warning: '+  dataset_root_path + "labelme_json/" + filestr + "/img.png" + '  ' + 'Not found')
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask ###重要
    def load_mask(self, image_id): # load_image_gt()中用到
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)##############
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)##############
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        labels = []
        labels = self.from_yaml_get_class(image_id)############

        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("Orange") != -1:
                #print(labels[i])
                labels_form.append("Orange")

        # sonclass good bad
        sonlabels_form = []
        for i in range(len(labels)):
            if labels[i].find("bad") != -1:#坏了
                sonlabels_form.append("bad")
            else:
                sonlabels_form.append("good")

        # grandclass ripe unripe
        grandlabels_form = []
        for i in range(len(labels)):
            if labels[i].find("unripe") != -1:#没熟
                grandlabels_form.append("unripe")
            elif labels[i].find("rare") != -1:#半熟
                grandlabels_form.append("rare")
            elif labels[i].find("bad") != -1:
                grandlabels_form.append("BG")
            else:
                grandlabels_form.append("ripe")


        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        sonclass_ids = np.array([self.sonclass_names.index(s) for s in sonlabels_form])
        grandclass_ids = np.array([self.grandclass_names.index(s) for s in grandlabels_form])

        #print('image_id:'+str(image_id))
        return mask, class_ids.astype(np.int32), sonclass_ids.astype(np.int32), grandclass_ids.astype(np.int32),

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# 基础设置
dataset_root_path="data/train/"
img_floder = dataset_root_path + "images"
mask_floder = dataset_root_path + "mask"
imglist = os.listdir(img_floder)
count = len(imglist)

dataset_train = DrugDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist,dataset_root_path)
dataset_train.prepare()
print("dataset_train-->",dataset_train._image_ids)


dataset_root_path="data/train/"
img_floder = dataset_root_path + "images"
mask_floder = dataset_root_path + "mask"
imglist = os.listdir(img_floder)
count = len(imglist)

dataset_val = DrugDataset() # val与train可以公用一个数据集,val的作用在于测试是否过拟合,以及来调整超参数
dataset_val.load_shapes(200, img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()
print("dataset_val-->",dataset_val._image_ids)


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    print("image_id",image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids, sonclass_ids, grandclass_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=400, layers='all')

