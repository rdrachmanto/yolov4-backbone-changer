import argparse
import numpy as np
import torch
from functools import partial

from nets.yolo_darknet import YoloDarknetBody
from nets.yolo import YoloBody
from torch.utils.data import DataLoader

from utils.utils import (get_anchors, get_classes, seed_everything,
                         show_config, worker_init_fn)
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.callbacks import EvalCallback, LossHistory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default=None, help='backbone for yolo')
    parser.add_argument('--model_path', type=str, default=None, help='model path')

    args = parser.parse_args()

    Cuda = True
    seed = 11
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    focal_loss          = False
    focal_alpha         = 0.25
    focal_gamma         = 2
    iou_type            = 'ciou'
    classes_path    = 'model_data/clp_classes.txt'
    anchors_path    = 'model_data/clp_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape     = [416, 416]
    batch_size = 1
    num_workers = 4

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    rank            = 0

    if args.backbone == None:
            raise ValueError("Select the backbone using flag --backbone")
    backbone        = args.backbone
    model_path      = args.model_path

    seed_everything(seed)

    val_annotation_path     = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    if backbone == 'cspdarknet53':
        model = YoloDarknetBody(anchors_mask, num_classes)
    else:
        model = YoloBody(anchors_mask, num_classes, backbone=backbone)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    else:
        raise ValueError('Please set the model path!')
    
    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing, focal_loss, focal_alpha, focal_gamma, iou_type)

    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_val     = len(val_lines)

    val_sampler     = None
    shuffle         = True
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = 600, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
    
    gen_val         = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    model.eval()
    for iteration, batch in enumerate(gen_val):
        images, targets = batch[0], batch[1]
        outputs = model(images)
