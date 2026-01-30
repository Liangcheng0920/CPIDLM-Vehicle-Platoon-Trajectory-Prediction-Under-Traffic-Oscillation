"""
Structural Agnostic Model (SAM)
基于原始模型结构的改进版本。

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018 (原始日期) | Revised: 2025
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class CNormalized_Linear(nn.Module):
    """
    带有列归一化的线性层。
    对应于原代码中对 self.weight 按照列进行归一化。
    """

    def __init__(self, in_features, out_features, bias=False):
        super(CNormalized_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # 对权重按列归一化（即对每个输入特征归一化）
        norm_weight = self.weight / self.weight.pow(2).sum(dim=0, keepdim=True).sqrt()
        return nn.functional.linear(input, norm_weight, self.bias)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None})")


class SAM_discriminator(nn.Module):
    """
    SAM 模型中的判别器。
    """

    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_discriminator, self).__init__()
        activation_function = kwargs.get('activation_function', nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.0)

        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if dropout != 0.0:
                layers.append(nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SAM_block(nn.Module):
    """
    SAM 的基本生成器模块（Block）。
    该模块使用 CNormalized_Linear 层和常规的线性层组合，并在输入端加入过滤器，
    用于屏蔽不需要连接的节点。
    """

    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_block, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        activation_function = kwargs.get('activation_function', nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalized_Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

        # 构造过滤器，用于屏蔽不应参与生成的节点
        self.register_buffer('_filter', th.ones(1, sizes[0]))
        for i in zero_components:
            self._filter[:, i] = 0.0
        self.fs_filter = nn.Parameter(self._filter.clone())
        if gpu:
            self._filter = self._filter.cuda(gpu_no)

    def forward(self, x):
        filtered_x = x * (self._filter * self.fs_filter).expand_as(x)
        return self.layers(filtered_x)


class SAM_generators(nn.Module):
    """
    所有生成器的集合。每个变量对应一个生成器块。
    """

    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        super(SAM_generators, self).__init__()
        rows, self.cols = data_shape
        if batch_size == -1:
            batch_size = rows
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)

        # 为每个生成器创建噪声向量（后续每次 forward 会更新）
        self.noise = [th.randn(batch_size, 1) for _ in range(self.cols)]
        if gpu:
            self.noise = [n.cuda(gpu_no) for n in self.noise]
        self.blocks = nn.ModuleList()
        for i in range(self.cols):
            self.blocks.append(SAM_block([self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        # 每次 forward 时重新生成噪声
        batch_size = x.size(0)
        self.noise = [th.randn(batch_size, 1, device=x.device) for _ in range(self.cols)]
        generated_variables = [self.blocks[i](th.cat([x, self.noise[i]], dim=1))
                               for i in range(self.cols)]
        return generated_variables


def run_SAM(df_data, skeleton=None, **kwargs):
    """
    执行 SAM 模型，估计变量间因果关系。
    参数：
      df_data: 包含数据的 pandas DataFrame 或 numpy 数组
      skeleton: 先验的因果结构（邻接矩阵），可选。如果提供，则利用其中的信息屏蔽某些连接。
      kwargs: 其他参数，如学习率、训练轮数等。
    返回：
      causal_filters: 估计得到的因果关系矩阵（numpy 数组）
    """
    gpu = kwargs.get('gpu', False)
    gpu_no = kwargs.get('gpu_no', 0)
    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)
    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    dnh = kwargs.get('dnh', None)
    plot = kwargs.get("plot", False)
    plot_generated_pair = kwargs.get("plot_generated_pair", False)

    # 数据处理：如果是 DataFrame 则转换为 numpy 数组
    if hasattr(df_data, 'columns'):
        list_nodes = list(df_data.columns)
        data_np = df_data[list_nodes].values.astype('float32')
    else:
        list_nodes = list(range(df_data.shape[1]))
        data_np = df_data.astype('float32')
    data_tensor = th.from_numpy(data_np)
    if batch_size == -1:
        batch_size = data_tensor.size(0)
    rows, cols = data_tensor.size()

    # 根据 skeleton 得到每个生成器屏蔽的节点信息
    if skeleton is not None:
        zero_components = [[] for _ in range(cols)]
        # 假设 skeleton 为二值矩阵，1 表示存在边，0 表示不存在边
        non_edges = (1 - skeleton)
        for i in range(non_edges.shape[0]):
            for j in range(non_edges.shape[1]):
                if non_edges[i, j] != 0:
                    zero_components[j].append(i)
    else:
        # 默认屏蔽本身（即自环）
        zero_components = [[i] for i in range(cols)]

    # 先复制一份 kwargs，并删除其中的 batch_size 参数
    kwargs_for_generators = kwargs.copy()
    kwargs_for_generators.pop("batch_size", None)

    sam = SAM_generators((rows, cols), zero_components, batch_size=batch_size,
                         batch_norm=True, **kwargs_for_generators)

    # 构造判别器，注意临时去掉 activation_function 参数以避免冲突
    activation_function = kwargs.get('activation_function', nn.Tanh)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("activation_function", None)
    discriminator_sam = SAM_discriminator([cols, dnh, dnh, 1], batch_norm=True,
                                          activation_function=nn.LeakyReLU,
                                          activation_argument=0.2, **kwargs_copy)

    # 将模型和数据移至 GPU（如果可用）
    if gpu:
        sam = sam.cuda(gpu_no)
        discriminator_sam = discriminator_sam.cuda(gpu_no)
        data_tensor = data_tensor.cuda(gpu_no)

    # 定义损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = optim.Adam(discriminator_sam.parameters(), lr=lr_disc)

    true_variable = th.ones(batch_size, 1, device=data_tensor.device)
    false_variable = th.zeros(batch_size, 1, device=data_tensor.device)
    causal_filters = th.zeros(cols, cols, device=data_tensor.device)

    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 用于绘图
    if plot:
        plt.ion()
        fig, ax = plt.subplots()
        adv_plt, gen_plt, l1_plt = [], [], []

    total_epochs = train_epochs + test_epochs
    for epoch in range(total_epochs):
        # 累计每个 epoch 的损失，用于监控训练进程
        epoch_adv_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_l1_loss = 0.0
        batch_count = 0

        for i_batch, (batch,) in enumerate(data_loader):
            batch_count += 1
            # 构造每个变量的列向量
            batch_vectors = [batch[:, i:i + 1] for i in range(cols)]

            current_batch_size = batch.size(0)
            true_variable = th.ones(current_batch_size, 1, device=data_tensor.device)
            false_variable = th.zeros(current_batch_size, 1, device=data_tensor.device)

            ##########################
            # 1. 更新判别器（Discriminator）
            ##########################
            d_optimizer.zero_grad()
            generated_variables = sam(batch)
            disc_losses = []
            for i in range(cols):
                generator_output = th.cat(batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:], dim=1)
                disc_output_detached = discriminator_sam(generator_output.detach())
                disc_loss_fake = criterion(disc_output_detached, false_variable)
                disc_losses.append(disc_loss_fake)
            true_output = discriminator_sam(batch)
            disc_loss_real = criterion(true_output, true_variable)
            adv_loss = (sum(disc_losses) / cols) + disc_loss_real

            adv_loss.backward()
            d_optimizer.step()

            ##########################
            # 2. 更新生成器（Generator）
            ##########################
            g_optimizer.zero_grad()
            generated_variables = sam(batch)
            gen_losses = []
            for i in range(cols):
                generator_output = th.cat(batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:], dim=1)
                disc_output = discriminator_sam(generator_output)
                gen_losses.append(criterion(disc_output, true_variable))
            gen_loss = sum(gen_losses)

            filters = th.stack([abs(block.fs_filter[0, :-1]) for block in sam.blocks], dim=1)
            l1_reg = regul_param * filters.sum()
            loss = gen_loss + l1_reg

            loss.backward()
            if epoch >= train_epochs:
                causal_filters += filters.detach()
            g_optimizer.step()

            # 累计每个 batch 的损失
            epoch_adv_loss += adv_loss.item()
            epoch_gen_loss += gen_loss.item()
            epoch_l1_loss += l1_reg.item()

            if plot and i_batch == 0:
                adv_plt.append(adv_loss.item())
                gen_plt.append((gen_loss.item() / cols))
                l1_plt.append(l1_reg.item())
                ax.clear()
                ax.plot(adv_plt, "r-", linewidth=1.5, label="Discriminator Loss")
                ax.plot(gen_plt, "g-", linewidth=1.5, label="Generator Loss")
                ax.plot(l1_plt, "b-", linewidth=1.5, label="L1 Regularization")
                ax.legend()
                plt.pause(0.001)

            if plot_generated_pair and epoch % 200 == 0:
                plt.figure()
                i, j = 0, 1  # 示例选择前两个变量
                plt.scatter(generated_variables[i].detach().cpu().numpy(),
                            batch[:, j].detach().cpu().numpy(), label="Y -> X")
                plt.scatter(batch[:, i].detach().cpu().numpy(),
                            generated_variables[j].detach().cpu().numpy(), label="X -> Y")
                plt.scatter(batch[:, i].detach().cpu().numpy(),
                            batch[:, j].detach().cpu().numpy(), label="Original Data")
                plt.legend()
                plt.show()

        # 计算并输出每个 epoch 的平均损失
        avg_adv = epoch_adv_loss / batch_count
        avg_gen = epoch_gen_loss / batch_count
        avg_l1 = epoch_l1_loss / batch_count
        if verbose:
            phase = "Train" if epoch < train_epochs else "Test"
            print(f"[{phase} Epoch {epoch+1}/{total_epochs}] "
                  f"Avg Discriminator Loss: {avg_adv:.4f}, "
                  f"Avg Generator Loss: {avg_gen:.4f}, "
                  f"Avg L1 Reg Loss: {avg_l1:.4f}")

    causal_filters = causal_filters / test_epochs
    return causal_filters.cpu().numpy()



class SAM(object):
    """
    结构无关模型（Structural Agnostic Model, SAM）
    用于通过对抗训练估计变量之间的因果关系。
    """

    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1):
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batchsize = batchsize

    def predict(self, data, skeleton=None, nruns=6, njobs=1, gpus=0, verbose=True,
                plot=False, plot_generated_pair=False, return_list_results=False):
        """
        对给定数据（及先验 skeleton，如有）执行 SAM 模型。

        参数：
          data: 包含观测数据的 pandas DataFrame 或 numpy 数组
          skeleton: 先验因果关系的邻接矩阵（可选），可为有向或无向
          nruns: 多次运行次数（推荐 ≥12 次以获得稳定结果）
          njobs: 并行运行的任务数（如无 GPU 则建议为 1，否则为 GPU 数量的两倍）
          gpus: 可用 GPU 数量
          verbose: 是否输出训练过程信息
          plot: 是否交互式绘制损失曲线（nruns 较大时不推荐）
          plot_generated_pair: 是否交互式绘制生成对（nruns 较大时不推荐）
          return_list_results: 是否返回每次运行的结果列表

        返回：
          如果 return_list_results 为 False，则返回平均后的因果关系矩阵；否则返回每次运行的结果列表。
        """
        results = Parallel(n_jobs=njobs)(
            delayed(run_SAM)(data, skeleton=skeleton,
                             lr_gen=self.lr, lr_disc=self.dlr,
                             regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                             gpu=bool(gpus), train_epochs=self.train_epochs,
                             test_epochs=self.test_epochs, batch_size=self.batchsize,
                             plot=plot, verbose=verbose, gpu_no=idx % max(gpus, 1))
            for idx in range(nruns)
        )
        if return_list_results:
            return results
        else:
            W = results[0]
            for w in results[1:]:
                W += w
            W /= nruns
            return W


if __name__ == "__main__":
    # # 示例用法
    # import numpy as np
    # import pandas as pd
    #
    # # 生成一个简单的合成数据集：100 个样本，4 个变量
    # np.random.seed(42)
    # data_np = np.random.randn(100, 4)
    # df_data = pd.DataFrame(data_np, columns=['X1', 'X2', 'X3', 'X4'])
    #
    # # 可选：提供先验 skeleton（例如，对角线为0，其他均为1，表示所有变量间均有联系）
    # skeleton = np.ones((4, 4)) - np.eye(4)
    #
    # sam_model = SAM(lr=0.01, dlr=0.01, l1=0.1, nh=50, dnh=50,
    #                 train_epochs=500, test_epochs=100, batchsize=32)
    #
    # causal_matrix = sam_model.predict(df_data, skeleton=skeleton,
    #                                   nruns=3, njobs=1, gpus=0,
    #                                   verbose=True, plot=False)
    # print("Estimated causal matrix:")
    # print(causal_matrix)
    torch.manual_seed(42)
    # 加载数据（请替换为实际数据文件路径）
    data = sio.loadmat('data_fine_0.1.mat')
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # (样本数, 时间步长, 特征数)

    # 选择需要的列（例如选取5列）并取最后50个时间步
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    train_real_speed1 = torch.tensor(data['lable_data'], dtype=torch.float32)
    train_real_speed = train_real_speed1[:, :, 0]  # 取车速（第一列）

    print(train_real_speed[1:100])
    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")

    # 单位转换（例如 ft/s 转为 m/s）
    train_data[:, :, 0] *= 0.3048
    train_data[:, :, 1] *= 0.3048
    train_data[:, :, 2] *= 0.3048
    train_data[:, :, 3] *= 0.3048
    train_data[:, :, 4] *= 0.3048
    train_real_speed *= 0.3048

    # 为演示取部分数据（例如取 10%）
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 0.01)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]

    # 数据检查
    # check_data(train_data, "train_data")
    # check_data(train_real_speed, "train_real_speed")

    # 提取安全距离：取 train_data 中最后一帧的第2列（索引1）
    train_s_safe = train_data[:, -1, 1].clone().detach()
    print("处理后时序示例（第一个样本）:")
    print(train_data[0, :, 0])
    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")
    print(f"train_s_safe shape: {train_s_safe.shape}")

    train_s_safe *= 0.3048

    # 划分训练集和测试集
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)
    train_data_part, test_data = train_data[:train_size], train_data[train_size:]
    train_real_speed_part, test_real_speed = train_real_speed[:train_size], train_real_speed[train_size:]
    train_s_safe, test_s_safe = train_s_safe[:train_size], train_s_safe[train_size:]

    # 创建数据加载器
    train_dataset = TensorDataset(train_data_part, train_real_speed_part, train_s_safe)
    test_dataset = TensorDataset(test_data, test_real_speed, test_s_safe)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #########################################
    # 用 SAM 模型构造因果关系邻接矩阵
    #########################################
    # 为 SAM 模型提供输入数据：取 train_data_part 最后时间步（形状：(样本数, 特征数)）
    sam_input = train_data_part[:, -1, :].cpu().numpy()
    print(sam_input.shape)
    sam_model = SAM(lr=0.1, dlr=0.1, l1=0.1, nh=20, dnh=20,
                    train_epochs=10, test_epochs=10, batchsize=32)
    # 这里多次运行以获得平均结果
    causal_matrix = sam_model.predict(sam_input, skeleton=None, nruns=6, njobs=1, gpus=0,
                                      verbose=True, plot=False, plot_generated_pair=False)
    print("SAM 估计的因果关系矩阵:")
    print(causal_matrix)

    # # 计算第一列和最后一列的最后5个值的方差
    # first_col_last_5 = batch_data[:, -5:, 0]  # (batch, 5)
    # last_col_last_5 = batch_data[:, -5:, -1]  # (batch, 5)
    # var_first_col = torch.var(first_col_last_5, dim=1)  # (batch,)
    # var_last_col = torch.var(last_col_last_5, dim=1)  # (batch,)
    #
    # # 合并这两列方差，作为orig_input
    # orig_input = torch.stack((var_first_col, var_last_col), dim=1)  # (batch, 2)