import argparse
from datetime import datetime
import numpy as np
import torch
from functools import partial

from nets.yolo_darknet import YoloDarknetBody
from nets.yolo import YoloBody
from torch.utils.data import DataLoader

from utils.monitor_thread import CPU, Memory, jstat_start, jstat_stop
from utils.utils import get_anchors, get_classes, seed_everything, worker_init_fn
from utils.dataloader import YoloDataset, yolo_dataset_collate


# global variables
# should be constants, set in other file!
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
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def setup_model(backbone, anchors_mask, num_classes, model_path):
    if backbone == 'cspdarknet53':
        model = YoloDarknetBody(anchors_mask, num_classes)
    else:
        model = YoloBody(anchors_mask, num_classes, backbone=backbone)

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

    model.to(device)

    return model


def load_dataset(val_annotation_path, input_shape, num_classes, shuffle=True, batch_size=1, num_workers=4):
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()

    val_sampler     = None
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = 600, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
    gen_val         = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    return gen_val


def infer(model, gen_val, is_warmup=False):
    torch.cuda.synchronize()

    model.eval()

    with torch.inference_mode():
        for iteration, batch in enumerate(gen_val):
            images, targets = batch[0], batch[1]
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            
            model(images)
            torch.cuda.synchronize()
            
            if is_warmup == True and iteration >= 10:
                break
            elif is_warmup == False and iteration >= 1000:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default=None, help='backbone for yolo', required=True)
    parser.add_argument('--model-path', type=str, default='', help='model path', required=True)
    parser.add_argument('--classes-path', type=str, default='', help='classes path', required=True)
    parser.add_argument('--anchors-path', type=str, default='', help='anchors path', required=True)
    parser.add_argument('--annot-path', type=str, default='', help='path of the set (train, test) for inference', required=True)

    args = parser.parse_args()
    print(args)

    classes_path    = args.classes_path
    anchors_path    = args.anchors_path
    input_shape     = [416, 416]
    batch_size = 1
    num_workers = 4

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    rank            = 0

    backbone        = args.backbone
    model_path      = args.model_path

    seed_everything(seed)

    val_annotation_path     = args.annot_path

    class_names, num_classes = get_classes(args.classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    model = setup_model(
        backbone=args.backbone,
        anchors_mask=anchors_mask,
        num_classes=num_classes,
        model_path=args.model_path
    )
    model.to(device)

    gen_val = load_dataset(
        val_annotation_path=args.annot_path,
        num_classes=num_classes,
        input_shape=[416, 416],
    )

    infer(model, gen_val, is_warmup=True)

    # Begin monitor thread
    cpu_thread = CPU()
    cpu_thread.start()
    mem_thread = Memory()
    mem_thread.start()
    jstat_start()

    # Inference
    start = datetime.now()
    infer(model, gen_val, is_warmup=False)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    lat = round((elapsed / 1000) * 1000, 2)

    print(f"Elapsed: {elapsed}s!")
    print(f"Latency: {lat}ms!")

    # Monitor thread stops
    cpu_thread.stop()
    mem_thread.stop()
    cpu_thread.join()
    mem_thread.join()

    jstat = jstat_stop()
    pow = float(jstat[1])
    gpu = float(jstat[0])

    cpu_use = round(cpu_thread.result[0], 2)  # type: ignore
    mem_use = round(mem_thread.result[0] / 1024, 2)  # type: ignore
    gpu = round(gpu, 2)

    print(f"CPU: {cpu_use}")
    print(f"GPU: {gpu}")
    print(f"Mem: {mem_use}")
    print(f"VDD: {pow}")
