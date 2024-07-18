# A script to visualize the ERF.
# Adapted from https://github.com/DingXiaoH/RepLKNet-pytorch
#
# Figure 1 in the paper used 120-epochs trained model with no EMA, to replicate it use:
# python erf.py --from-erf-matrix outputs_erf/pre_computed/RepLK-T_31.npy
# python erf.py --from-erf-matrix outputs_erf/pre_computed/SLaK-T_51.npy
# python erf.py --from-erf-matrix outputs_erf/pre_computed/WTConvNeXt-T_5.npy
# 
# Alternatively, you can run other trained models, e.g., 300-epochs wtconvnext-t with ema, using:
# python erf.py --data-dir IMAGENET_PATH --model wtconvnext_tiny --checkpoint WTConvNeXt_tiny_5_300e_ema.pth 
#
# --------------------------------------------------------'
import os
import argparse
import numpy as np
import torch
from timm.utils.metrics import AverageMeter

from torchvision import datasets, transforms
from torch import optim as optim

from timm.models import create_model, load_checkpoint

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns

import wtconvnext

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

plt.rcParams["font.family"] = "Times New Roman"
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
sns.set_style("white")
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--data-dir', type=str, help='dataset path')
    parser.add_argument('--model', default=None, type=str, help='dataset path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint file.')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use (default: cuda)')
    parser.add_argument('--num-images', default=50, type=int, help='Num of images to use')
    parser.add_argument('--save-erf-matrix', default=None, type=str, help='path to the contribution score matrix (.npy file)')
    parser.add_argument('--from-erf-matrix', default=None, type=str, help='analyze from processed erf matrix.')
    parser.add_argument('--output-file', default='outputs_erf/heatmap.png', type=str, help='where to save the heatmap')
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model.forward_features(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def process(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_dir)
    root = os.path.join(args.data_dir, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    sampler_val = torch.utils.data.RandomSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)


    print("Creating model")

    model = create_model(
        args.model, 
        pretrained=False, 
        num_classes=1000,
        )
    
    load_checkpoint(model, args.checkpoint)
    print(f"loaded pretrained model from {args.checkpoint}")

    model.to(args.device)
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in enumerate(data_loader_val):

        if meter.count == args.num_images:
            return meter.avg

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)

def heatmap(data, camp='coolwarm', figsize=(10, 10), ax=None, save_path=None):
    plt.figure(figsize=figsize, dpi=40)
    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap=camp, square=True, vmin=0, vmax=1,
                annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
    plt.savefig(save_path)


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None


def analyze_erf(data, output_file):
    print(np.max(data))
    print(np.min(data))
    data = np.log10(data + 1)       #   the scores differ in magnitude. take the logarithm for better readability
    data = data / np.max(data)      #   rescale to [0,1] for the comparability among models
    print('======================= the high-contribution area ratio =====================')
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        side_length, area_ratio = get_rectangle(data, thresh)
        print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
    heatmap(data, save_path=output_file)
    print('heatmap saved at ', output_file)


if __name__ == '__main__':
    args = parse_args()
    if args.from_erf_matrix is None:
        erf_matrix = process(args)
    else:
        erf_matrix = np.load(args.from_erf_matrix)

    analyze_erf(erf_matrix, args.output_file)
