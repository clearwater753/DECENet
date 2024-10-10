import os
import numpy as np
from PIL import Image
from collections import Counter

def calculate_class_weights(label_dir, num_classes, ignore_label=255):
    pixel_counts = Counter()
    total_pixels = 0

    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        label = np.array(Image.open(label_path))
        
        # 忽略标签值为 ignore_label 的像素
        mask = label != ignore_label
        label = label[mask]
        
        pixel_counts.update(label.flatten())
        total_pixels += label.size

    class_weights = []
    for i in range(num_classes):
        if i in pixel_counts:
            class_weight = total_pixels / (num_classes * pixel_counts[i])
        else:
            class_weight = 0.0
        class_weights.append(class_weight)

    return class_weights

# 类别数为 19
label_dir = '/root/autodl-tmp/bdd100k/labels/sem_seg/masks/train'
num_classes = 19
class_weights = calculate_class_weights(label_dir, num_classes)

print("Class weights:", class_weights)

