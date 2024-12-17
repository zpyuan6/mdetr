import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours

from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import os
import yaml
import time
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score

torch.set_grad_enabled(False)

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



def model_prediction(img, caption):

    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)


    outputs['probas'] = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = outputs['probas'] > 0.7
    outputs['probas'] = outputs['probas'][keep]

    # probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()

    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()

    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]

    # Check labels
    if len(labels) != outputs['probas'].shape[0]:
        target_label_num = outputs['probas'].shape[0]
        check_item = list(predicted_spans .keys())
        for i in range(target_label_num):
            if i not in check_item:
                labels.insert(i, '')

    outputs['labels'] = [l.replace(' ','') for l in labels]

    outputs['pred_boxes'] = outputs['pred_boxes'].cpu()[0, keep].numpy()
    
    return outputs


def load_dataset_yaml(dataset_yaml_path: str = "coco8.yaml") -> list:
    """Load categories names from a dataset YAML file in YOLO farmat."""
    with open(dataset_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def load_image_and_labels(image_path: str, label_path: str):
    """Load an image and its corresponding labels."""
    image = Image.open(image_path).convert("RGB")
    with open(label_path, 'r') as file:
        raw_labels = [line.strip().split() for line in file.readlines()]

        labels = []

        for raw_label in raw_labels:
            if len(raw_label) == 5:
                category, cx, cy, w, h = raw_label
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                labels.append({
                    'category_id': int(category),
                    'bbox': [cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h]
                    })


    return image, labels

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    pcx, pcy, pw, ph = box1
    x1g, y1g, x2g, y2g = box2

    x1 = pcx - 0.5 * pw
    y1 = pcy - 0.5 * ph
    x2 = pcx + 0.5 * pw
    y2 = pcy + 0.5 * ph

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def asign_label_id(text1):
    """Check if two texts are similar."""
    if 'DITY' in text1.upper():
        return 5    
    elif 'CY' in text1.upper():
        if 'J' in text1.upper():
            return 4
        else:
            return 1
    elif 'PRAT' in text1.upper():
        return 3
    elif 'GL' in text1.upper():
        return 2
    elif 'ME' in text1.upper():
        return 0

    return -1

def model_validation(
    data: dict,
    model: torch.nn.Module,
    save_path: str = None):

    if save_path is None:
        save_path = "results"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    categories = []

    for i, category in data.get("names").items():
        categories.append(category)

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dir = data['val']
    label_dir = image_dir.replace('images', 'labels')

    image_files = [f for f in os.listdir(os.path.join(data['path'], image_dir)) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG') or f.endswith('.jpeg') or f.endswith('.JPEG')]
    
    results = []
    all_labels = []
    all_scores = []
    all_ious = []

    t = 0
    for image_file in tqdm(image_files):
        image_path = os.path.join(data['path'], image_dir, image_file)
        file_name = ".".join(image_file.split(".")[:-1])
        label_path = os.path.join(data['path'], label_dir, f"{file_name}.txt")

        image, labels = load_image_and_labels(image_path, label_path)
        image = transform(image).unsqueeze(0).cuda()

        caption = ". ".join(categories)

        start_time = time.time()
        outputs = model_prediction(image, caption)
        end_time = time.time()
        t += (end_time - start_time)

        pboxes = outputs['pred_boxes'].tolist()
        plabel = outputs['labels']
        pscore = outputs['probas'].tolist()

        for i in range(len(pboxes)):
            results.append({
                'image_id': image_file,
                'category_id': plabel[i],
                'bbox': pboxes[i],
                'score': pscore[i]
                })
            all_scores.append(pscore[i])
            all_labels.append(plabel[i])
            ious = [compute_iou(pboxes[i], l['bbox']) for l in labels if l['category_id'] == asign_label_id(plabel[i])]

            all_ious.append(max(ious) if ious else 0)

    detection_speed = t / len(image_files)
    
    # Save results to file
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    # Print detection speed
    print(f"Detection speed: {detection_speed} seconds per image")

    iou_thresholds = list(np.arange(0.5, 0.95, 0.05))

    aps = []
    for iou_thresh in iou_thresholds:
        y_true = [1 if iou >= iou_thresh else 0 for iou in all_ious]
        ap = average_precision_score(y_true, all_scores)
        print(f"AP at IoU {iou_thresh}: {ap}")
        aps.append(ap)

    print(f"mAP: {np.mean(np.array(aps))}")




if __name__ == "__main__":

    data_yaml_file = "ppn_zero_shot_0_1_val.yaml"
    # Load the dataset YAML file
    data = load_dataset_yaml(data_yaml_file)

    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()
    model.eval()

    model_validation(data, model)

