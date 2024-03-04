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
def measure_latency(model, device='cpu', num_trials=5):
    print("Measuring latency...")
    input_data = torch.randn(1, 3, 416, 416).to(device)
    print('Input size:', input_data.shape)
    model = model.to(device)

    latencies = []
    for i in range(num_trials):
        latency = benchmark.Timer(
            stmt='model(input_data)',
            globals={'model': model, 'input_data': input_data},
            num_threads=1,
        ).blocked_autorange(min_run_time=1).median * 1000  # Konversi dari detik ke milidetik(ms)
        latencies.append(latency)
        print(f"Trial {i+1}: {latency:.4f} ms")

    average_latency = sum(latencies) / num_trials
    print(f"Average latency over {num_trials} trials: {average_latency:.4f} ms")
    return latencies, average_latency

if __name__ == "__main__":
    model = load_yolov4_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_trials = 5  # Ubah sesuai kebutuhan
    trial_latencies, average_latency = measure_latency(model, device=device, num_trials=num_trials)
    print(f"Latensi model YOLOv4 di perangkat {device}: {average_latency:.4f} ms (rata-rata dari {num_trials} trials)")
