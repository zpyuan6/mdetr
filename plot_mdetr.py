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

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def apply_mask(image, mask, color=0, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img).transpose(1, 2, 0)
    ax = plt.gca()
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask in zip(scores, boxes.tolist(), labels, masks):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none")
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

def model_prediction(img, caption):

    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)


    outputs['probas'] = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = outputs['probas'] > 0.7
    outputs['probas'] = outputs['probas'][keep]

    # probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()

    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.01).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]

    outputs['labels'] = [l.replace(' ','') for l in labels]

    outputs['pred_boxes'] = outputs['pred_boxes'].cpu()[0, keep]

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'], img.shape[-2:])

    plot_results(img.cpu().squeeze(), outputs['probas'], bboxes_scaled, outputs['labels'])
    



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

def model_run(
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

        
    detection_speed = t / len(image_files)
    





if __name__ == "__main__":

    data_yaml_file = "ppn_zero_shot_0_1_val.yaml"
    # Load the dataset YAML file
    data = load_dataset_yaml(data_yaml_file)

    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()
    model.eval()

    model_run(data, model)

