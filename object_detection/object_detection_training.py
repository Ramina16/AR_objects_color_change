
import cv2
import os

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from super_gradients.training.datasets import YoloDarknetFormatDetectionDataset
from super_gradients.training.transforms.transforms import (
    DetectionRandomAffine,
    DetectionHSV,
    DetectionHorizontalFlip,
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionTargetsFormatTransform
)
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models

from object_detection.constants import CLASS_NAMES, NUM_CLASSES, DATASET_PATH
from object_detection.utils import get_class_colors, get_prediction_from_model, plot_prediction


class_colors = get_class_colors(CLASS_NAMES)


if __name__ == '__main__':
    train_dataset = YoloDarknetFormatDetectionDataset(data_dir=DATASET_PATH, images_dir=os.path.join(DATASET_PATH, 'train', 'image'), 
                                                      labels_dir=os.path.join(DATASET_PATH, 'train', 'labels'), classes=CLASS_NAMES,
                                                      input_dim=(640, 640), ignore_empty_annotations=True, 
                                                      transforms=[
        DetectionRandomAffine(degrees=0.0, scales=(0.9, 1.1), shear=0.0, target_size=(640, 640), filter_box_candidates=True, border_value=128),
        DetectionHorizontalFlip(prob=0.5),
        DetectionHSV(prob=1.0, hgain=5, vgain=50, sgain=50),
        DetectionPaddedRescale(input_dim=(640, 640)),
        DetectionStandardize(max_value=255),
        DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH")
    ])
    val_dataset = YoloDarknetFormatDetectionDataset(data_dir=DATASET_PATH, images_dir=os.path.join(DATASET_PATH, 'val', 'image'), 
                                                    labels_dir=os.path.join(DATASET_PATH, 'val', 'labels'), classes=CLASS_NAMES,
                                                    input_dim=(640, 640), ignore_empty_annotations=True,
                                                    transforms=[
        DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
        DetectionStandardize(max_value=255),
        DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH"),
    ])
    
    test_dataset = YoloDarknetFormatDetectionDataset(data_dir=DATASET_PATH, images_dir=os.path.join(DATASET_PATH, 'test', 'image'), 
                                                     labels_dir=os.path.join(DATASET_PATH, 'test', 'labels'), classes=CLASS_NAMES,
                                                     input_dim=(640, 640), ignore_empty_annotations=True, 
                                                     transforms=[
                                                         DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH"),
                                                         DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
        DetectionStandardize(max_value=255),])
    
    NUM_WORKERS = 2
    BATCH_SIZE = 16

    train_dataloader_params = {
        "shuffle": True,
        "batch_size": BATCH_SIZE,
        "drop_last": True,
        "pin_memory": True,
        "collate_fn": DetectionCollateFN(),
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
    }

    val_dataloader_params = {
        "shuffle": False,
        "batch_size": BATCH_SIZE,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": DetectionCollateFN(),
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
    }
    
    test_dataloader_params = {
        "shuffle": False,
        "batch_size": BATCH_SIZE,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": DetectionCollateFN(),
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_dataset, **train_dataloader_params)
    valid_loader = DataLoader(val_dataset, **val_dataloader_params)
    test_loader = DataLoader(test_dataset, **test_dataloader_params)
    
    
    train_params = {
        "warmup_initial_lr": 1e-5,
        "initial_lr": 3e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.5,
        "optimizer": "AdamW",
        "zero_weight_decay_on_bias_and_bn": True,
        "lr_warmup_epochs": 6,
        "warmup_mode": "LinearEpochLRWarmup",
        "optimizer_params": {"weight_decay": 0.0001},
        "ema": False,
        "average_best_models": False,
        "ema_params": {"beta": 25, "decay_type": "exp"},
        "max_epochs": 50,
        "mixed_precision": True,
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=NUM_CLASSES, reg_max=16),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=NUM_CLASSES,
                normalize_targets=True,
                include_classwise_ap=True,
                class_names=CLASS_NAMES,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
            )
        ],
        "metric_to_watch": "mAP@0.50",
    }
    
    
    trainer = Trainer(experiment_name="yolo_nas_s_indoor", ckpt_root_dir="yolo-experiments")
    model = models.get(Models.YOLO_NAS_S, num_classes=NUM_CLASSES, pretrained_weights="coco")
        
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)
    best_model = models.get(Models.YOLO_NAS_S, num_classes=NUM_CLASSES, 
                            checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))
    # best_model = models.get(Models.YOLO_NAS_S, num_classes=NUM_CLASSES, 
    #                         checkpoint_path='/home/olena/projects/AR_objects_color_change/object_detection/' \
    #                         'yolo-experiments/yolo_nas_s_indoor/RUN_20250317_035330_189454/ckpt_best.pth')
    # test_metrics = trainer.test(model=best_model, test_loader=test_loader, test_metrics_list=[
    #         DetectionMetrics_050(
    #             score_thres=0.1,
    #             top_k_predictions=300,
    #             num_cls=NUM_CLASSES,
    #             normalize_targets=True,
    #             include_classwise_ap=True,
    #             class_names=CLASS_NAMES,
    #             post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
    #         )
    #     ])
    
    os.makedirs(os.path.join(trainer.checkpoints_dir_path, 'pred'), exist_ok=True)
    for img_name in test_dataset.images_file_names[:20]:
        image_path = os.path.join(test_dataset.images_folder, img_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes, class_names, class_indices, confidences = get_prediction_from_model(img, best_model)
        
        img = plot_prediction(img, class_names, bboxes, class_indices, confidences, class_colors)
        
        plt.imsave(os.path.join(trainer.checkpoints_dir_path, 'pred', img_name, img))
        

