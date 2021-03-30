import argparse
import torch
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.datasets.kitti.eval import get_official_eval_result
from det3d.models import build_detector
from det3d.torchie.parallel import MegDataParallel
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device

import matplotlib.pyplot as plt
from pydash import at
from pathlib import Path
import numpy as np
import math

class_to_name = {
    0: "car",
    1: "pedestrian",
    2: "bicycle",
    3: "truck",
    4: "bus",
    5: "trailer",
    6: "construction_vehicle",
    7: "motorcycle",
    8: "barrier",
    9: "traffic_cone",
    10: "cyclist",
}

def drawBoundingBoxes(ax, x, y, z, w, l, h, r, col='b', linewidth=2):
    """
    Draws bounding boxe lines to given axis
    Params:
        (x, y, z): center point coordinates of an object 
        (w, l, h): width, length and height of the bounding box / object
        r: rotation in radians
    """
    # Do this, because we have center point
    l = l / 2.0
    w = w / 2.0

    # Calculate corner locations with rotation
    x1 = x + (w * math.cos(r) + l * math.sin(r))
    y1 = y + (-w * math.sin(r) + l * math.cos(r))
    x2 = x + (-w * math.cos(r) + l * math.sin(r))
    y2 = y + (+w * math.sin(r) + l * math.cos(r))
    x3 = x + (-w * math.cos(r) - l * math.sin(r))
    y3 = y + (w * math.sin(r) - l * math.cos(r))
    x4 = x + (w * math.cos(r) - l * math.sin(r))
    y4 = y + (-w * math.sin(r) - l * math.cos(r))

    # Bottom rectangle
    ax.plot3D([x3, x4], [y3, y4], [z, z], col, linewidth=2, alpha=0.8)
    ax.plot3D([x2, x1], [y2, y1], [z, z], col, linewidth=2, alpha=0.8)
    ax.plot3D([x3, x2], [y3, y2], [z, z], col, linewidth=2, alpha=0.8)
    ax.plot3D([x4, x1], [y4, y1], [z, z], col, linewidth=2, alpha=0.8)

    # Top rectangle
    ax.plot3D([x3, x4], [y3, y4], [z+h, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x2, x1], [y2, y1], [z+h, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x3, x2], [y3, y2], [z+h, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x4, x1], [y4, y1], [z+h, z+h], col, linewidth=2, alpha=0.8)

    # Vertical lines
    ax.plot3D([x1, x1], [y1, y1], [z, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x2, x2], [y2, y2], [z, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x3, x3], [y3, y3], [z, z+h], col, linewidth=2, alpha=0.8)
    ax.plot3D([x4, x4], [y4, y4], [z, z+h], col, linewidth=2, alpha=0.8)


def init_plot():
    """
    Cretes a new plot with settings given below
    Return: axes for subplots in the figure
    """
    fig = plt.figure(constrained_layout=True, figsize=(7,9), dpi=130)
    gs = fig.add_gridspec(5, 1)
    ax2 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :], projection='3d')

    tick_color = (0.2, 0.2, 0.2, 1.0)
    pane_color = (0.12, 0.12, 0.12, 1.0)
    ax1.w_xaxis.set_pane_color(pane_color)
    ax1.w_yaxis.set_pane_color(pane_color)
    ax1.w_zaxis.set_pane_color(pane_color)

    ax1.tick_params(axis='x', colors=tick_color)
    ax1.tick_params(axis='y', colors=tick_color)
    ax1.tick_params(axis='z', colors=tick_color)
    ax1.view_init(elev=90, azim=180)

    ax1.set_xlim3d(0, 80)
    ax1.set_zlim3d(-2, 5)
    
    return (ax1, ax2)

def plot_image(ax, example, training=True):
    """
    Plots image and shows it on given axis. Only plots the first image
    in batch. Thus, the batch size should be set to 1 when creating plots.
    Params:
        ax: figure axis
        example: dictionary that has been moved to target device
    """
    ax.grid(False) # hide grid lines
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if training:
        prefix = example['metadata'][0]['image_prefix'] / "training/image_2/"
    else:
        prefix = example['metadata'][0]['image_prefix'] / "testing/image_2/"

    image_file = prefix / Path(f"{example['metadata'][0]['image_idx']:06d}" + ".png")
    img = plt.imread(image_file)
    ax.imshow(img)

def draw_annotations(ax, example, annotations_num, show_label=True):
    """
    Draws annotated bounding boxes.
    Params:
        ax: axis where boxes will be drawn
        example: dict that contains annotated data
        annotations_num: number of annotations (number of boxes)
        show_label: should annotation class be shown in visualization?
    """
    for n in range(annotations_num):
        anno = example['annos'][n]
        for q, category in enumerate(anno['names']):
            x, y, z, w, l, h, r = example['annos'][n]['boxes'][q]
            drawBoundingBoxes(ax, x, y, z, w, l, h, r, col='red')

            if show_label:
                ax.text(x, y, z+h, f"{category}", color='r', fontsize=8.0, rotation=math.degrees(r))

def draw_predictions(ax, outputs):
    """
    Draws predicted bounding boxes.
    Params:
        ax: axis where boxes will be drawn.
        outputs (dict): model outputs when example has been fed in.
    """
    for output in outputs:
        boxes = output['box3d_lidar'].cpu().detach().numpy()
        confidences = output['scores'].cpu().detach().numpy()
        classes = output['label_preds'].cpu().detach().numpy()
        class_txts = at(class_to_name, *classes)
        for k, box3d in enumerate(boxes):
            x, y, z, w, l, h, r = box3d
            drawBoundingBoxes(ax, x, y, z, w, l, h, r, col='green', linewidth=0.8)
            ax.text(x+(w/2.0)+1, y+(l/2.0)+2, z+h, f"{class_txts[k]}<{confidences[k]:.2f}>", color=(0.4, 0.95, 0.3), fontsize=8.0, rotation=math.degrees(r))

def draw_pointcloud(ax, example):
    """
    Draws the lidar point cloud (raw data).
    Params:
        ax: axis where points will be drawn.
        example (dict): dictionary that includes pointcloud points
    """
    points = example['points'].cpu().detach().numpy()
    points_num = len(points)
    xs = np.empty([points_num])
    ys = np.empty([points_num])
    zs = np.empty([points_num])
    intensity = np.empty([len(points)])
    for j, point in enumerate(points):
        xs[j] = point[1]
        ys[j] = point[2]
        zs[j] = point[3]
        intensity[j] = point[4]

    intensity = intensity
    ax.scatter3D(xs, ys, zs, c=intensity, marker='.', s=0.3, cmap=plt.get_cmap('jet'))

def visualize(model, data_loader, device, timer=None, show=False):
    model.eval()

    for i, batch in enumerate(data_loader):
        example = example_to_device(batch, device=device)
        with torch.no_grad():

            ax1, ax2 = init_plot()

            draw_pointcloud(ax1, example)

            plot_image(ax2, example)

            draw_annotations(ax1, example, len(batch['metadata']))

            outputs = model(example, return_loss=False, rescale=not show)

            draw_predictions(ax1, outputs) 

            plt.show(block=True)

    return results_dict


def main():
    # Usage: python visualize.py <config_path> <checkpoint_path>
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    args = parser.parse_args()

    cfg = torchie.Config.fromfile(args.config_path)
    cfg.data.val.test_mode = True 

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint_path, map_location="cpu")
    model = MegDataParallel(model, device_ids=[0])

    device = torch.device("cuda")
    visualize(model, data_loader, device)

if __name__ == "__main__":
    main()
