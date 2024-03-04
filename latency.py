import torch
import torch.utils.benchmark as benchmark
from nets.yolo_darknet import YoloDarknetBody
from nets.yolo import YoloBody
import argparse
import time

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 4
pretrained = False

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default=None, help='backbone for yolo')
args = parser.parse_args()

# Fungsi untuk membuat dan memuat model YOLOv4
def load_yolov4_model():
    if args.backbone == 'cspdarknet53':
        model = YoloDarknetBody(anchors_mask, num_classes, pretrained = pretrained)
    else:
        model = YoloBody(anchors_mask, num_classes, backbone=args.backbone, pretrained = pretrained)
    model.eval()  
    return model

# Fungsi untuk melakukan inferensi pada model dan mengukur latensi
def measure_latency(model, device='cpu'):
    print("Measuring latency...")
    input_data = torch.randn(1, 3, 416, 416).to(device)
    print('Input size:', input_data.shape)
    model = model.to(device)

    latencies = benchmark.Timer(
        stmt='model(input_data)',
        globals={'model': model, 'input_data': input_data},
        num_threads=1,
    ).blocked_autorange(min_run_time=1)

    return latencies.median * 1000  # Konversi dari detik ke milidetik(ms)

if __name__ == "__main__":
    model = load_yolov4_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latency = measure_latency(model, device=device)
    print(f"Latensi model YOLOv4 di perangkat {device}: {latency:.4f} ms")
