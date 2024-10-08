"""Main training script for synthetic problems."""

import argparse
import os
import time
import scipy.stats as st
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import higher

import layers
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder

TRAIN_BATCH = 16 # 32
TEST_BATCH = 4 # 10


def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter, wandb=0):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
                net,
                inner_opt,
                copy_initial_weights=False,
                override=inner_opt_builder.overrides,
        ) as (
                fnet,
                diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()

    metrics = {"train_loss": np.mean(qry_losses)}
    if wandb:
        wandb.log(metrics, step=step_idx)
    meta_opt.step()


def train_sparsity(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter, lam, wandb=0):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)
    dims = (x_spt[0].size(dim=2), y_spt[0].size(dim=2))

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    meta_opt.zero_grad()

    for i in range(task_num):
        with higher.innerloop_ctx(
                net,
                inner_opt,
                copy_initial_weights=False,
                override=inner_opt_builder.overrides,
        ) as (
                fnet,
                diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()

    metrics = {"train_loss": np.mean(qry_losses)}
    if wandb:
        wandb.log(metrics, step=step_idx)
    meta_opt.step()

    for name, param in net.named_parameters():
        if name[-4:] == 'warp':
            with torch.no_grad():
                param.copy_(exclusive_reg(param, dims, lam, 'feature'))


def group_reg(param, dims, lam, group='feature'):
    n, m = dims
    k = param.size(dim=1)
    if group == 'feature':
        proj_factor = (1 - lam / torch.norm(param.view(n, -1, k), p=2, dim=0).view(1, m, k).repeat(n, 1, 1))
        return (param.view(n, -1, k) * proj_factor).view(-1, k)
    elif group == 'cofeature':
        proj_factor = (1 - lam / torch.norm(param.view(-1, m, k), p=2, dim=1).view(n, 1, k).repeat(1, m, 1))
        return (param.view(-1, m, k) * proj_factor).view(-1, k)


def exclusive_reg(param, dims, lam, group='feature'):
    n, m = dims
    k = param.size(dim=1)
    if group == 'feature':
        proj_factor = torch.norm(param.view(n, -1, k), p=1, dim=0).repeat(n, 1, 1).view(-1, k)
        return torch.sign(param) * torch.relu(torch.abs(param) - lam * proj_factor)
    elif group == 'cofeature':
        proj_factor = torch.norm(param.view(-1, m, k), p=1, dim=1).repeat(1, m, 1).view(-1, k)
        return torch.sign(param) * torch.relu(torch.abs(param) - lam * proj_factor)


def tradeoff_reg():
    return


def test(step_idx, data, net, inner_opt_builder, n_inner_iter, wandb=0):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    for i in range(task_num):
        with higher.innerloop_ctx(
                net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
                fnet,
                diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    if wandb:
        wandb.log(test_metrics, step=step_idx)
    return avg_qry_loss


working_path = "D:\Dataset\meta_learning_symmetries"


class PathFileNames:
    def __init__(self, args):
        self.problem = args.problem
        self.lam_reg = args.lam_reg
        self.ntasks = args.ntasks
        self.model = args.model
        self.group_name = "{}-{}".format(args.problem, args.model)
        self.epochs = args.num_outer_steps

    """
    def get_model_path(self):
        return os.path.join(os.path.dirname(__file__), os.path.join('outputs', 'synthetic_outputs', \
                                                                    'sparsity_model', 'models'))
    """

    def get_model_file_name(self):
        model_file_name = "{}_{}-{}_{}-{}_".format(
                                                            "lam", str(self.lam_reg), "epochs",
                                                            str(self.epochs), "version")
        return model_file_name

    """
    {}-{}_{}-{}_{}-{}_{}-{}_
    self.group_name, "ntasks",
                                                            str(self.ntasks), \
                                                            """

    def get_base_path(self, mode='base'):
        path = os.path.join(os.path.dirname(__file__), os.path.join('outputs', 'synthetic_outputs'))
        if mode == 'base':
            path = os.path.join(path, 'base_model')
        elif mode == 'sparsity':
            path = os.path.join(path, 'sparsity_model')
        return path

    def get_extention_path(self):
        return os.path.join(self.group_name, 'ntasks' + str(self.ntasks))

    def get_visual_out_path(self, mode='base'):
        path = self.get_base_path(mode)
        path = os.path.join(path, 'weight_visualization', self.get_extention_path())
        return path

    def get_model_file_path(self, mode='base'):
        model_file_dir = os.path.join(self.get_base_path(mode), 'models', self.get_extention_path())
        model_file_name = self.get_model_file_name()
        files = [f for f in os.listdir(model_file_dir) if os.path.isfile(os.path.join(model_file_dir, f))]
        version = []
        for f in files:
            if f[:f.rfind('_') + 1] == model_file_name:
                version.append(f[f.rfind('_') + 1:-4])
        """
        for i in range(len(version)):
            print("({})".format(i + 1) + " : version " + version[i])
        answer = input("Chosen version : ")
        assert (answer.isdigit())
        answer = int(answer)
        assert (1 <= answer <= len(version))
        """
        answer = 1
        model_file_name = model_file_name + str(version[answer - 1]) + '.pth'
        return os.path.join(model_file_dir, model_file_name), version[answer - 1]

    def set_model_file_path(self, mode='base'):
        model_file_dir = os.path.join(self.get_base_path(mode), 'models', self.get_extention_path())
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir)
        model_file_name = self.get_model_file_name()
        files = [f for f in os.listdir(model_file_dir) if os.path.isfile(os.path.join(model_file_dir, f))]
        version = []
        for f in files:
            if f[:f.rfind('_') + 1] == model_file_name:
                version.append(f[f.rfind('_') + 1:-4])
        version = [int(v) for v in version]
        if version == []:
            model_version = 0
        else:
            model_version = max(version) + 1
        model_file_name += str(model_version) + '.pth'
        return os.path.join(model_file_dir, model_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--lam_reg", type=float, default=0.001)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ntasks", type=str, default=100)  #
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--trainer", type=str, default='base')

    args = parser.parse_args()
    path_name = PathFileNames(args)
    if args.trainer == 'base':
        mode = 'base'
        project_name = "reparametrization"
    else:
        mode = 'sparsity'
        project_name = "reparametrization_sparsity"
    group_name = "{}-{}".format(args.problem, args.model)
    if args.wandb:
        wandb_run = wandb.init(project=project_name, group=group_name,
                               dir=os.path.join(path_name.get_base_path(mode=mode), 'models'))
        wandb.config.update(args)
    device = torch.device(args.device)
    db = SyntheticLoader(device, problem=args.problem, ntasks=args.ntasks, k_spt=args.k_spt, k_qry=args.k_qry)

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

    inner_opt_builder = InnerOptBuilder(
        net, device, args.inner_opt, args.init_inner_lr, "learned", args.lr_mode
    )
    if args.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=args.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=args.outer_lr)

    start_time = time.time()
    for step_idx in range(args.num_outer_steps):
        data, _filters = db.next(TRAIN_BATCH, "train")
        if mode == 'base':
            train(step_idx, data, net, inner_opt_builder, meta_opt, args.num_inner_steps, args.wandb)
        else:
            train_sparsity(step_idx, data, net, inner_opt_builder, meta_opt, args.num_inner_steps, args.lam_reg,
                           args.wandb)
        if step_idx == 0 or (step_idx + 1) % 100 == 0:
            test_data, _filters = db.next(TEST_BATCH, "test")  #
            val_loss = test(
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                args.num_inner_steps,
            )
            if step_idx > 0:
                steps_p_sec = (step_idx + 1) / (time.time() - start_time)
                if args.wandb:
                    wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
                # print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}")

    # save final model
    model_file_path = path_name.set_model_file_path(mode=mode)
    torch.save(net.state_dict(), model_file_path)
    if args.wandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_file_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # Test loss result
    path_name = PathFileNames(args)
    model_file_path, version = path_name.get_model_file_path(mode=mode)
    visual_out_path = path_name.get_visual_out_path(mode=mode)
    visual_file_name = path_name.get_model_file_name() + str(version)
    val_loss = test(-1, test_data, net, inner_opt_builder, args.num_inner_steps)
    if not os.path.exists(os.path.join(visual_out_path, "loss")):
        os.makedirs(os.path.join(visual_out_path, "loss"))
    with open(os.path.join(visual_out_path, "loss", visual_file_name + '.txt'), 'w') as f:
        f.write("Test Loss : " + str(val_loss))


if __name__ == "__main__":
    main()
