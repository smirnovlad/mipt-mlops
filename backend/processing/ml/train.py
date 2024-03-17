import gc
from pathlib import Path
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from roboflow import Roboflow
from torch import optim
from torch.utils.data import DataLoader

from .segmentation.SegNet import SegNet

IMAGE_HEIGHT, IMAGE_WIDTH = 1280, 768

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Train. Device: {device}")


def load_dataset():
    rf = Roboflow(api_key="2NBbupbg2nhidokX1UR0")
    project = rf.workspace("study-jzyvf").project("metro-detection")
    project.version(6).download("coco-segmentation")


def load_annotations():
    coco_annotations_path = "./Metro-Detection-6/train/_annotations.coco.json"
    coco = COCO(coco_annotations_path)

    image_ids = coco.getImgIds(imgIds=coco.getImgIds())
    images_folder_path = "./Metro-Detection-6/train"

    images = []
    masks = []

    for img_id in image_ids:
        img_info = coco.loadImgs(ids=img_id)[0]
        image_path = f"{images_folder_path}/{img_info['file_name']}"
        image = plt.imread(image_path)

        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        annotations = coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in annotations:
            coco_mask = coco.annToMask(ann)
            mask = np.maximum(mask, coco_mask * ann['category_id'])

        image = extend_image(image, 3)
        mask = extend_image(mask)

        images.append(image)
        masks.append(mask)

    return images, masks


def extend_image(img, channels=None):
    height, width = img.shape[0], img.shape[1]
    delta = IMAGE_WIDTH - width
    if channels:
        padding = np.zeros((height, int(delta / 2), channels), np.uint8)
    else:
        padding = np.zeros((height, int(delta / 2)), np.uint8)
    img = np.concatenate((padding, img, padding), axis=1)
    return img


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, limit=0.5):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = torch.sigmoid(outputs.squeeze(1)) > limit  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    smooth = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds

    return thresholded


def bce(y_pred, y_real):
    return y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))


def bce_loss(y_pred, y_real):
    return torch.mean(bce(y_pred, y_real))


def score_model(model, metric, data, threshold=0.5):
    model.eval()  # testing mode
    scores = 0
    total_size = 0
    with torch.set_grad_enabled(False):
        for X_batch, Y_label in data:
            total_size += len(X_batch)
            Y_pred = model(X_batch.to(device))
            scores += metric(Y_pred, Y_label.to(device), limit=threshold).mean().item() * len(X_batch)

    return scores / total_size


def fit_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    processed_data = 0

    tic = time()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)

        del inputs
        del outputs
        del labels
        torch.cuda.empty_cache()
        gc.collect()

    toc = time()

    print("Fit epoch time: ", toc - tic)
    train_loss = running_loss / processed_data
    return train_loss


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    tic = time()
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)

        del inputs
        del outputs
        del labels
        torch.cuda.empty_cache()
        gc.collect()

    toc = time()

    print("Eval epoch time: ", toc - tic)

    val_loss = running_loss / processed_size
    return val_loss


def train(model, opt, loss_fn, epochs, data_tr, data_val, scheduler=None):
    print('qq')
    torch.autograd.set_detect_anomaly(True)
    loss_history = []
    score_history = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()

        print('* Epoch %d/%d' % (epoch + 1, epochs))

        train_loss = fit_epoch(model, data_tr, loss_fn, opt)
        val_loss = eval_epoch(model, data_val, loss_fn)
        if scheduler:
            scheduler.step(val_loss)

        loss_history.append((train_loss, val_loss))

        # train_score = score_model(model, iou_pytorch, data_tr)
        tic = time()
        val_score = score_model(model, iou_pytorch, data_val)
        toc = time()
        print("Score on eval time: ", toc - tic)
        score_history.append(val_score)

        print(f'\n\tTrain loss: {train_loss};'
              f'\n\tVal loss: {val_loss};'
              f' \n\tVal score: {val_score};')

    return loss_history, score_history


def start_training(message):
    print(message)

    torch.manual_seed(42)
    np.random.seed(42)

    load_dataset()
    images, masks = load_annotations()

    DATA_SIZE = len(images)
    ix = np.random.choice(len(images), DATA_SIZE, False)
    tr, val = np.split(ix, [int(0.9 * DATA_SIZE)])

    images = np.array(images)
    masks = np.array(masks)

    batch_size = 3
    data_tr = DataLoader(list(zip(np.rollaxis(images[tr], 3, 1), masks[tr, np.newaxis])),
                         batch_size=batch_size, shuffle=True)
    data_val = DataLoader(list(zip(np.rollaxis(images[val], 3, 1), masks[val, np.newaxis])),
                          batch_size=batch_size, shuffle=False)

    segnet_model_bce = SegNet().to(device)

    optimizer = optim.AdamW(segnet_model_bce.parameters(), lr=0.0001)
    loss_hist_1, score_hist_1 = train(segnet_model_bce, optimizer, bce_loss, 30, data_tr, data_val)

    optimizer = optim.AdamW(segnet_model_bce.parameters(), lr=0.000025)
    loss_hist_2, score_hist_2 = train(segnet_model_bce, optimizer, bce_loss, 5, data_tr, data_val)

    optimizer = optim.AdamW(segnet_model_bce.parameters(), lr=0.00001)
    loss_hist_3, score_hist_3 = train(segnet_model_bce, optimizer, bce_loss, 5, data_tr, data_val)

    optimizer = optim.AdamW(segnet_model_bce.parameters(), lr=0.000002)
    loss_hist_4, score_hist_4 = train(segnet_model_bce, optimizer, bce_loss, 5, data_tr, data_val)

    torch.save(segnet_model_bce.state_dict(), f"segnet_bce_1125_{45}_epoch.pth")
