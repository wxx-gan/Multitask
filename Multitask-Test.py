# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
from tqdm import tqdm
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#from samples.coco import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "weigts")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0350.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/test/images/")

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
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 576

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (6 * 6, 12 * 6, 24 * 6, 48 * 6, 96 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 30

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Orange']
sonclass_names = ['BG','Bad', 'Good']
grandclass_names = ['BG','UnRipe','Rare','Ripe']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

for i in tqdm(file_names):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, i))
    print(str(i))
    a=datetime.now()

    # Run detection
    results = model.detect([image], verbose=1)

    b=datetime.now()
    # Visualize results
    print("Time:",(b-a).seconds)
    r = results[0]
    visualize.display_instances_wcx(i,
                                image,
                                r['rois'],  #box
                                r['masks'],  #mask
                                r['class_ids'],  #class_id
                                class_names,  # class_names = ['BG', 'orange']

                                r['sonclass_ids'],  # class_id
                                sonclass_names,  # sonclass_names = ['bad', 'good']

                                r['grandclass_ids'],  # class_id
                                grandclass_names,  # grandclass_names = ['unripe', 'ripe']

                                r['scores'],
                                r['sonscores'],
                                r['grandscores'],

                                show_mask=False)
