from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train
from detectron2.data.datasets import register_coco_instances

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=32,
    num_workers=18,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="maize_valid", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=18,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


def register_dataset(path_to_data):
    register_coco_instances("maize_train", { },
                            path_to_data + "/coco_annotations/train_2022.json",
                            path_to_data + "/data")
    register_coco_instances("maize_valid", { },
                            path_to_data + "/coco_annotations/valid_2022.json",
                            path_to_data + "/data")


path_to_data = '/home/appuser/data/maize_dataset'
register_dataset(path_to_data)

dataloader.train.mapper.use_instance_mask = False
# dataloader.train.dataset.names = 'maize_train'
# dataloader.test.dataset.names = 'maize_valid'
# dataloader.train.total_batch_size = 4

optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
model.pixel_mean = [136.25, 137.81, 135.14]
model.num_classes = 2

train.output_dir = '/home/appuser/logs'
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
