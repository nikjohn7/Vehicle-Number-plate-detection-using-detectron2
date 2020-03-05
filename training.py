import matplotlib.pyplot as plt
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow

# detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

# register dataset
register_coco_instances("plates", {}, "./plates_coco/annotations.json", "./plates_coco/")

plates_metadata = MetadataCatalog.get("plates")
dataset_dicts = DatasetCatalog.get("plates")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import torch, os

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml")


cfg.DATASETS.TRAIN = ("plates",)
cfg.DATASETS.TEST = ()   
cfg.DATALOADER.NUM_WORKERS = 2

# initialize model from model zoo
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  
#cfg.MODEL.WEIGHTS = 'output/model_final.pth' 
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.MAX_ITER = 10000    
cfg.SOLVER.CHECKPOINT_PERIOD = 500

# we've only one class - plate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()