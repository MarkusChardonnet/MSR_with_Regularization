import argparse
import os
import time
import wandb
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder
import train_synthetic

import layers

from os import listdir
from os.path import isfile, join
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from train_synthetic import PathFileNames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ntasks", type=str, default=None)
    parser.add_argument("--lam_reg", type=float, default=0.001)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--trainer", type=str, default='base')

    args = parser.parse_args()
    group_name = "{}-{}".format(args.problem, args.model)
    device = torch.device(args.device)
    if args.trainer == 'base':
        mode = 'base'
    else:
        mode = 'sparsity'

    if args.problem in ["2d_rot8_flip", "2d_rot8", "2d_rot4"]:
        if args.problem == "2d_rot8":  # C_8, 8 elements
            c_o = 24
        elif args.problem == "2d_rot8_flip":  # D_8, 16 elements
            c_o = 48
        elif args.problem == "2d_rot4":  # C_4, 4 elements
            c_o = 12
        else:
            raise NotImplementedError

        if args.model == "share_conv":
            net = nn.Sequential(layers.ShareConv2d(1, c_o, 3, bias=False)).to(device)
        elif args.model == "conv":
            net = nn.Sequential(nn.Conv2d(1, c_o, 3, bias=False)).to(device)
        elif args.model == "share_fc":
            net = nn.Sequential(layers.ShareLinearFull(70, 68, bias=False, latent_size=c_o)).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")
    elif args.problem in ["rank1", "rank2", "rank5", "rank2_kernel5", "rank5_kernel5"]:
        if args.model == "lc":
            net = nn.Sequential(layers.LocallyConnected1d(1, 1, 68, kernel_size=3, bias=False)).to(
                device
            )
        elif args.model == "fc":
            net = nn.Sequential(nn.Linear(70, 68, bias=False)).to(device)
        elif args.model == "conv":
            net = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, bias=False)).to(device)
        elif args.model == "share_fc":
            latent = {"rank1": 3, "rank2": 6, "rank5": 30, "rank2_kernel5": 6, "rank5_kernel5": 30}[args.problem]
            in_features = 70
            if args.problem in ["rank2_kernel5", "rank5_kernel5"]:
                in_features = 72
            net = nn.Sequential(layers.ShareLinearFull(in_features, 68, bias=False, latent_size=latent)).to(
                device
            )
        elif args.model == "share_conv":
            tmp = {"rank1": 3, "rank2": 6, "rank5": 30}[args.problem]
            net = nn.Sequential(layers.ShareConv2d(1, tmp, 3, bias=False)).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")

    path_name = PathFileNames(args)
    model_file_path, version = path_name.get_model_file_path(mode=mode)
    net.load_state_dict(torch.load(model_file_path))
    visual_out_path = path_name.get_visual_out_path(mode=mode)
    visual_file_name = path_name.get_model_file_name() + str(version)

    is_warp = False
    warp_params = []
    # print("Parameter tensors :")
    # print()
    for k in net.state_dict().keys():
        # print("Parameter : ", k)
        # print("Shape : ", net.state_dict()[k].detach().cpu().shape)
        # print()
        if k[-4:] == "warp":
            is_warp = True
            warp_params.append(net.state_dict()[k].detach().cpu().numpy())

    # Weight Value Histogram
    if is_warp:
        # print("Model is shared. ")
        for w in warp_params:
            _ = plt.hist(w, bins=20, range=(0., 1.))  # arguments are passed to np.histogram
            plt.title("Symmetry Matrix absolute Weights Histogram ")
            if not os.path.exists(os.path.join(visual_out_path, "hist_dist")):
                os.makedirs(os.path.join(visual_out_path, "hist_dist"))
            plt.savefig(os.path.join(visual_out_path, "hist_dist", visual_file_name + '.png'))
            plt.close()

    # Symmetry Matrix Visulaization
    for i in range(len(net)):
        if isinstance(net[0], layers.ShareLinearFull):
            in_features = net[0].in_features
            out_features = net[0].out_features
            x = np.max(np.abs(net[0].warp.view(out_features,in_features,-1).detach().cpu().numpy()),axis=2)
            plt.imshow(x, cmap='viridis', interpolation='nearest')
            plt.title("Symmetry Matrix absolute Weights (maximum over filter size) ")
            plt.colorbar()
            if not os.path.exists(os.path.join(visual_out_path, "heat_sym_matrix")):
                os.makedirs(os.path.join(visual_out_path, "heat_sym_matrix"))
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(visual_out_path, "heat_sym_matrix", visual_file_name + '.png'))


"""
def params_histogram(params_tensor):
    # np.histogram(np.abs(params_tensor), bins=50, range=(0., 1.))
    _ = plt.hist(params_tensor, bins=20, range=(0., 1.))  # arguments are passed to np.histogram
    plt.title("Histogram with weights absolute value ")
    plt.show()
    """


if __name__ == "__main__":
    main()
