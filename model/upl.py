from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer

import torch_points_kernels as tp
from util.logger import get_logger
from lib.pointops2.functions import pointops

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import matplotlib

class DPR(nn.Module):
    def __init__(self, feat_dim=192, attn_dropout=0.1,fixed_num_points=2048):
        super(DPR, self).__init__()
        self.in_channel = self.out_channel = feat_dim

        self.temperature = (self.out_channel ** 0.5)
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = fixed_num_points//4
        self.q_map = nn.Conv1d(fixed_num_points, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(fixed_num_points, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):
        assert not torch.isnan(query).any(), "query contains NaN"
        assert not torch.isnan(support).any(), "support contains NaN"
        assert not torch.isnan(prototype).any(), "prototype contains NaN"

        # query.shape: torch.Size([2, 192, 2048])
        # support.shape: torch.Size([2, 192, 2048])
        # prototype.shape: torch.Size([2, 3, 192])

        batch, dim = query.shape[0], query.shape[1]  # batch: 批大小, dim: 特征维度 1,192,2048
        way = support.shape[0] + 1  # way: 类别数
        residual = prototype  # residual: (batch, dim)
        q = self.q_map(query.transpose(1, 2))  # q.shape after q_map: torch.Size([2, 512, 192])
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)  # support.shape after cat: torch.Size([3, 192, 2048])
        k = self.k_map(support.transpose(1, 2))  # k: k.shape after k_map: torch.Size([[512, 576]])
        v = self.v_map(prototype)  # v.shape after v_map: torch.Size([2, 3, 192])
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])  # q.shape after view: torch.Size([512, 384])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])  # k.shape after view: torch.Size([512, 576])

        # 检查NaN
        assert not torch.isnan(q).any(), "q contains NaN"
        assert not torch.isnan(k).any(), "k contains NaN"
        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)#attn.shape after matmul: torch.Size([384, 576])
        assert not torch.isnan(attn).any(), "attn contains NaN before softmax"

        attn = attn.reshape(batch, way, dim, dim)  # attn.shape after reshape: torch.Size([2, 3, 192, 192])
        # attn = attn - attn.max(dim=-1, keepdim=True)[0]
        # 数值稳定处理

        # attn = torch.clamp(attn, min=-20, max=20)
        attn = F.softmax(attn, dim=-1)  # attn.shape after softmax: torch.Size([2, 3, 192, 192])
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

        assert not torch.isnan(attn).any(), "attn contains NaN after softmax"

        v = v.unsqueeze(2)  #v.shape after unsqueeze: torch.Size([2, 3, 1, 192])
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)  # output: (batch, dim, way+1)2, 192, 3
        output = self.dropout(self.fc(output)).transpose(1, 2)  # output: (batch, way, dim)[2, 3, 192]
        output = self.layer_norm(output + residual)  # output: (batch, way, dim)2, 3, 192]

        return  output
class UPL(nn.Module):
    """
    UPL: Few-shot点云分割主模型
    包含原型提取、原型变分推理、特征聚合、分类等模块
    """
    def __init__(self, args):
        super(UPL, self).__init__()
        # 基本参数初始化
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        self.base_proto_ema = getattr(args, "base_proto_ema", 0.999)  # 新增，默认值0.999

        # 损失函数定义
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(
            ignore_index=args.ignore_label
        )

        # Patch、窗口等参数预处理
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2**i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2**i) for i in range(args.num_layers)
        ]

        # 数据集相关参数
        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1, 3: 2, 4: 3, 8: 4, 10: 5, 11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 10: 6, 12: 7, 13: 8, 14: 9, 19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 4: 2, 8: 3, 9: 4, 11: 5, 15: 6, 16: 7, 17: 8, 18: 9, 20: 10,
                }

        # 日志
        if self.main_process():
            self.logger = get_logger(args.save_path)

        # 主干特征提取网络
        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.args.classes // 2 + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path),
        )

        self.feat_dim = args.channels[2]
        self.visualization = args.vis
        # ablation 开关
        self.use_vpir = getattr(args, "use_vpir", True)
        self.pa_type = getattr(args, "pa_type", "dpr").lower()

        # 原型相关层
        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes + 1, self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.kpconv = KPConvResBlock(
            self.feat_dim, self.feat_dim, 0.04, sigma=2
        )
        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )
        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim // 2, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        # 聚合层数
        if self.args.data_name == "s3dis":
            agglayers = 2
        else:
            agglayers = 4
        print(f"use agglayers {agglayers}")
        self.agglayers = nn.ModuleList(
            [
                AggregatorLayer(
                    hidden_dim=self.feat_dim,
                    guidance_dim=0,
                    nheads=4,
                    attention_type="linear",
                )
                for _ in range(agglayers)
            ]
        )

        # 类别降维
        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * (self.n_way + 1), self.feat_dim
            )

        # 背景原型降维
        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )
        self.max_kl_weight = getattr(args, "max_kl_weight", 1.0)
        # 推理采样次数
        self.num_inference_samples = getattr(args, 'num_inference_samples', 3)
        # 原型变换器（Transformer）
        self.fixed_num_points = getattr(args, "fixed_num_points", 2048)

        # 统一使用 DPR 模块；在 forward 中通过不同输入实现 dual-stream 或 single-stream
        self.transformer = DPR(feat_dim=self.feat_dim, fixed_num_points=self.fixed_num_points)
        
        # 读取配置文件中的hidden_dim参数
        var_infer_hidden_dim = getattr(args, "var_infer_hidden_dim", self.feat_dim // 2)

        # 先验分布参数网络（用于变分推理）
        self.prior_mu_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, var_infer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_infer_hidden_dim, self.feat_dim)
        )
        self.prior_sigma_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, var_infer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_infer_hidden_dim, self.feat_dim),
            nn.Softplus() 
        )

        # 后验分布参数网络
        self.posterior_mu_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, var_infer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_infer_hidden_dim, self.feat_dim)
        )
        self.posterior_sigma_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, var_infer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_infer_hidden_dim, self.feat_dim),
            nn.Softplus()
        )
        self.init_weights()

        # 基础类别原型缓存
        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )


        # 注意力机制相关
        self.bg_attn_mlp = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, 1),
        )
        self.fg_attn_mlp = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, 1),
        )


        # 添加KL损失warmup参数
        self.kl_warmup_epochs = getattr(args, "kl_warmup_epochs", 10)

    def init_weights(self):
        """
        初始化权重（除base_merge外）
        """
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        """
        判断是否为主进程（用于分布式训练日志等）
        """
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def get_kl_weight(self, epoch):
        """计算KL损失权重"""
        if epoch < self.kl_warmup_epochs:
            return self.max_kl_weight * (epoch / self.kl_warmup_epochs)
        return self.max_kl_weight

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:

        """
        Forward pass of the UPL model.

        Args:
            support_offset: Offset of each scene in the support set (shape: [N_way*K_shot]).
            support_x: Support point cloud inputs (shape: [N_support, in_channels]).
            support_y: Support masks (shape: [N_support]).
            query_offset: Offset of each scene in the query set (shape: [N_way]).
            query_x: Query point cloud inputs (shape: [N_query, in_channels]).
            query_y: Query labels (shape: [N_query]).
            epoch: Current training epoch.
            support_base_y: Base class labels in the support set (shape: [N_support]).
            query_base_y: Base class labels in the query set (shape: [N_query]).
            sampled_classes: The classes sampled in the current episode (shape: [N_way]).

        Returns:
            final_pred: Predicted class logits for query point clouds (shape: [1, n_way+1, N_query]).
            loss: The total loss value for this forward pass.
            avg_uncertainty: Mean uncertainty value across query points (float).
            kl_loss_val: KL loss value for this forward (torch.Tensor).
        """

        # get downsampled support features
        (
            support_feat,  # N_s, C
            support_x_low,  # N_s, 3
            support_offset_low,
            support_y_low,  # N_s
            _,
            support_base_y,  # N_s
        ) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y
        )
        assert support_y_low.shape[0] == support_x_low.shape[0]
        # 按offset分割特征
        support_offset_low = support_offset_low[:-1].long().cpu()
        support_feat_list = torch.tensor_split(support_feat, support_offset_low)
        # support_feat = [tensor.clone() for tensor in support_feat_list]  # 创建深拷贝
        support_feat = torch.tensor_split(support_feat, support_offset_low)
        support_x_low = torch.tensor_split(support_x_low, support_offset_low)
        if support_base_y is not None:
            support_base_y = torch.tensor_split(support_base_y, support_offset_low)
      
        fixed_support_feat = []
        for feat in support_feat_list:
            n_points = feat.shape[0]
            feat = feat.transpose(0, 1)
            if n_points >= self.fixed_num_points:
                idx = torch.randperm(n_points)[:self.fixed_num_points]
                feat_fixed = feat[:, idx]
            else:
                # 先采样所有点，再随机补充到2048个
                idx_all = torch.arange(n_points)
                idx_extra = torch.randint(0, n_points, (self.fixed_num_points - n_points,))
                idx = torch.cat([idx_all, idx_extra])
                feat_fixed = feat[:, idx]
            fixed_support_feat.append(feat_fixed)
        support_feat_init = torch.stack(fixed_support_feat)
        support_feat_init = support_feat_init.view(self.n_way, self.k_shot, -1, self.fixed_num_points)

        # 2. 支持集原型生成（前景/背景）
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset_low)
        bg_mask = torch.tensor_split(bg_mask, support_offset_low)

        # For k_shot, extract N_pt/k_shot per shot
        fg_prototypes = self.getPrototypes(
            support_x_low,
            support_feat,
            fg_mask,
            k=self.n_subprototypes // self.k_shot,
        )
        fg_prototypes_reshape = fg_prototypes.view(
            self.n_way, -1, self.feat_dim
        )
        bg_prototype = self.getPrototypes(
            support_x_low,
            support_feat,
            bg_mask,
            k=self.n_subprototypes // self.k_shot,
        )
        # 背景原型降维
        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(
                bg_prototype.permute(1, 0)
            ).permute(1, 0)
        bg_prototype_reduced = bg_prototype.clone()

        # 3. 查询集特征与原型处理
        (
            query_feat,  # N_q, C
            query_x_low,  # N_q, 3
            query_offset_low,
            query_y_low,  # N_q
            q_base_pred,  # N_q, N_base_classes
            query_base_y,  # N_q
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # split query features into list according to offset
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat_list = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        # query_feat = [tensor.clone() for tensor in query_feat_list]  # 创建深拷贝
        query_x_low_list = torch.tensor_split(query_x_low, query_offset_low_cpu)
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(query_base_y, query_offset_low_cpu)
        
        
        fixed_query_feat = []
        for feat in query_feat_list:  # feat: [N_pt, C]best_
            n_points = feat.shape[0]
            feat = feat.transpose(0, 1)  # [C, N_pt]
            if n_points > self.fixed_num_points:
                idx = torch.randperm(n_points)[:self.fixed_num_points]
                feat_fixed = feat[:, idx]
            elif n_points < self.fixed_num_points:
                # 先采样所有点，再补齐
                idx_all = torch.arange(n_points)
                idx_extra = torch.randint(0, n_points, (self.fixed_num_points - n_points,))
                idx = torch.cat([idx_all, idx_extra])
                feat_fixed = feat[:, idx]
            else:
                feat_fixed = feat

            fixed_query_feat.append(feat_fixed)
        query_feat_init = torch.stack(fixed_query_feat)  

        # 查询集原型生成（仅 DPR 使用；SPR 不使用 query 的 mask，避免泄露）
        if self.pa_type != "spr":
            q_fg_mask = query_y_low
            q_bg_mask = torch.logical_not(query_y_low)
            q_fg_mask = torch.tensor_split(q_fg_mask, query_offset_low_cpu)
            q_bg_mask = torch.tensor_split(q_bg_mask, query_offset_low_cpu)

            # For k_shot, extract N_pt/k_shot per shot
            q_fg_prototypes = self.query_getPrototypes(
                query_x_low_list,
                query_feat,
                q_fg_mask,
                k=self.n_subprototypes // self.k_shot,
            )
            q_fg_prototypes_reshape = q_fg_prototypes.view(
                self.n_way, -1, self.feat_dim
            )
            q_bg_prototype = self.query_getPrototypes(
                query_x_low_list,
                query_feat,
                q_bg_mask,
                k=self.n_subprototypes // self.k_shot,
            )
            if q_bg_prototype.shape[0] > self.n_subprototypes:
                q_bg_prototype = self.bg_proto_reduce(
                    q_bg_prototype.permute(1, 0)
                ).permute(1, 0)
            q_bg_prototype_reduced = q_bg_prototype.clone()

        # 4. 原型池化与拼接
        s_bg_proto_out = bg_prototype_reduced.mean(dim=0).unsqueeze(0)
        s_fg_proto_out = fg_prototypes_reshape.mean(dim=1)
        support_prototypes = torch.cat([s_bg_proto_out, s_fg_proto_out], dim=0) #2 way :3,192
        support_prototypes_all = support_prototypes.unsqueeze(0).repeat(query_feat_init.shape[0], 1, 1)#2 way :3,192
        support_feat_init_mean = support_feat_init.mean(1)
        # SPR 与 DPR 均使用 query_feat_init 作为 q 源；SPR 仅不构建 query 原型/不计算 posterior
        s_refined_prototypes = self.transformer(query_feat_init, support_feat_init_mean, support_prototypes_all) # 1way :1,2,192 2way: 2 3 192
        if self.pa_type != "spr":
            q_bg_proto_out = q_bg_prototype_reduced.mean(dim=0).unsqueeze(0)
            q_fg_proto_out = q_fg_prototypes_reshape.mean(dim=1)
            query_prototypes = torch.cat([q_bg_proto_out, q_fg_proto_out], dim=0) #2 way :3,192
            query_prototypes_all = query_prototypes.unsqueeze(0).repeat(query_feat_init.shape[0], 1, 1)#2 way :3,192
            q_refined_prototypes = self.transformer(query_feat_init, support_feat_init_mean, query_prototypes_all)# 1way :1,2,192 2way: 2 3 192
        else:
            q_refined_prototypes = s_refined_prototypes
        # 获取前景和背景的refined结果
        refined__bg = s_refined_prototypes[:, 0, :]  # 背景refined原型 (batch, 192)
        refined__fg = s_refined_prototypes[:, 1:, :]  # 前景refined原型 (batch, n_way, 192) 1,1,192

        # 5. 变分推理（原型采样）
        kl_losses = []
        if self.use_vpir:
            prior_mu = self.prior_mu_mlp(s_refined_prototypes)
            prior_sigma = self.prior_sigma_mlp(s_refined_prototypes)
            # 限制sigma的范围，避免数值不稳定
            prior_sigma = torch.clamp(prior_sigma, min=1e-6, max=10.0)
            if self.training and self.pa_type != "spr":
                posterior_mu = self.posterior_mu_mlp(q_refined_prototypes)
                posterior_sigma = self.posterior_sigma_mlp(q_refined_prototypes)
                posterior_sigma = torch.clamp(posterior_sigma, min=1e-6, max=10.0)
                # KL散度计算
                kl_div = (
                    torch.log(prior_sigma / (posterior_sigma + 1e-8)) +
                    (posterior_sigma.pow(2) + (posterior_mu - prior_mu).pow(2)) / (2.0 * prior_sigma.pow(2)) - 
                    0.5)
                kl_losses.append(kl_div.mean())

        # 基础类别原型更新（EMA）
        if self.training:
            for base_feat, base_y in zip(
                list(query_feat) + list(support_feat),
                list(query_base_y_list) + list(support_base_y),
            ):
                cur_baseclsses = base_y.unique()
                cur_baseclsses = cur_baseclsses[cur_baseclsses != 0]
                for class_label in cur_baseclsses:
                    class_mask = base_y == class_label
                    class_features = (
                        base_feat[class_mask].sum(dim=0) / class_mask.sum()
                    ).detach()
                    if torch.all(self.base_prototypes[class_label - 1] == 0):
                        self.base_prototypes[class_label - 1] = class_features
                    else:
                        self.base_prototypes[class_label - 1] = (
                            self.base_prototypes[class_label - 1] * self.base_proto_ema + class_features * (1 - self.base_proto_ema)
                        )
            # 当前episode目标类别不参与背景
            mask_list = [
                self.base_class_to_pred_label[base_cls] - 1
                for base_cls in sampled_classes
            ]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool
            )
            base_mask[mask_list] = False
            base_avail_pts = self.base_prototypes[base_mask]
            assert len(base_avail_pts) == self.base_classes - self.n_way
        else:
            base_avail_pts = self.base_prototypes

        query_pred = []
        num_samples = 1 if self.training else self.num_inference_samples
        # 6. 查询集推理与分类
        all_uncertainties = []
        for i, q_feat in enumerate(query_feat):
            if epoch < 1:
                base_guidance = None
            else:
                base_similarity = F.cosine_similarity(
                    q_feat[:, None, :],
                    base_avail_pts[None, :, :],
                    dim=2,
                )
                # max similarity for each query point as the base guidance
                base_guidance = base_similarity.max(dim=1, keepdim=True)[
                    0
                ]
                # 针对第i个batch构建sparse_embeddings
            sample_predictions = []
            final_pred_sample_list=[]
                
            for sample_idx in range(num_samples):
                if self.use_vpir:
                    eps = torch.randn_like(prior_mu)
                    refined_prototypes_vi = prior_mu + prior_sigma * eps
                else:
                    refined_prototypes_vi = s_refined_prototypes
                refined_prototypes_vi_i = refined_prototypes_vi[i]  # 只取当前batch
                refined__bg_vi = refined_prototypes_vi_i[0, :]      # refined__bg 1,192 bg_prototype_reduced,200,192
                refined__fg_vi = refined_prototypes_vi_i[1:, :]     # refined__fg_vi 1,1,192 fg_prototypes_reshape 1,200,192
                
                refined__bg_vi_expanded = refined__bg_vi.expand(bg_prototype_reduced.shape[0], -1)  # [N_bg, C]
                # 拼接背景原型：原始多个原型 + refined单个原型
                refined_bg_single = refined__bg[i].unsqueeze(0)  # (1, 192)
                bg_prototype_extended = torch.cat([bg_prototype_reduced, refined_bg_single], dim=0)  # [N_bg+1, 192]

                refined__bg_vi_expanded = refined__bg_vi.expand(bg_prototype_extended.shape[0], -1)  # [N_bg+1, C]
                attn_bg = torch.sigmoid(self.bg_attn_mlp(torch.cat([bg_prototype_extended, refined__bg_vi_expanded], dim=-1)))  # [N_bg+1, 2C]
                bg_prototype_new = (1 - attn_bg) * bg_prototype_extended + attn_bg * refined__bg_vi_expanded

                fg_prototypes_new_list = []
                for way_idx in range(self.n_way):
                    # 拼接前景原型：原始多个原型 + refined单个原型  
                    refined_fg_single = refined__fg[i, way_idx].unsqueeze(0)  # (1, 192)
                    fg_prototype_extended = torch.cat([fg_prototypes_reshape[way_idx], refined_fg_single], dim=0)  # [N_fg+1, 192]
                    
                    refined_fg_expanded = refined__fg_vi[way_idx].expand(fg_prototype_extended.shape[0], -1)  # [N_fg+1, C]
                    attn_fg = torch.sigmoid(
                        self.fg_attn_mlp(
                            torch.cat([fg_prototype_extended, refined_fg_expanded], dim=-1)
                        )
                    )  # [N_fg+1, 1]
                    fg_prototype_new = (1 - attn_fg) * fg_prototype_extended + attn_fg * refined_fg_expanded  # [N_fg+1, C]
                    fg_prototypes_new_list.append(fg_prototype_new)
                fg_prototypes_new_flat = torch.cat(fg_prototypes_new_list, dim=0)  # [n_way*N_fg, C]
                sparse_embeddings = torch.cat([bg_prototype_new, fg_prototypes_new_flat], dim=0)  # [*, C]
                # 假设 refined__fg 是不加噪声的 refined 前景原型，形状 [batch, n_way, C]

                # fg_prototypes_new_list = []
                # for way_idx in range(self.n_way):
                #     # 扩展 refined__fg 到每个前景原型的数量
                #     refined_fg_expanded = refined__fg[i, way_idx].expand(fg_prototypes_reshape[way_idx].shape[0], -1)  # [N_fg, C]
                #     attn_fg = torch.sigmoid(
                #         self.fg_attn_mlp(
                #             torch.cat([fg_prototypes_reshape[way_idx], refined_fg_expanded], dim=-1)
                #         )
                #     )  # [N_fg, 1]
                #     fg_prototype_new = attn_fg * fg_prototypes_reshape[way_idx] + (1-attn_fg) * refined_fg_expanded  # [N_fg, C]
                #     fg_prototypes_new_list.append(fg_prototype_new)
                # fg_prototypes_new_flat = torch.cat(fg_prototypes_new_list, dim=0)  # [n_way*N_fg, C]
                # sparse_embeddings = torch.cat([bg_prototype_new, fg_prototypes_new_flat], dim=0)  # [*, C]

                correlations = F.cosine_similarity(
                    q_feat[:, None, :],
                    sparse_embeddings[None, :, :],  
                    dim=2,
                ) 
                correlations = (
                    self.lin1(correlations.view(correlations.shape[0], self.n_way + 1, -1 ))
                    .permute(2, 1, 0)
                    .unsqueeze(0)
                )  

                for layer in self.agglayers:
                    correlations = layer(
                        correlations, base_guidance
                    )  # 1, C, N_way+1, N_q

                correlations = (
                    correlations.squeeze(0).permute(2, 1, 0).contiguous()
                )  # N_q, N_way+1, C

                # reduce the class dimension
                if self.n_way == 1:
                    correlations = self.class_reduce(correlations).squeeze(
                        1
                    )  # N_q, C
                else:
                    correlations = self.class_reduce(
                        correlations.view(correlations.shape[0], -1)
                    )  # N_q, C


                # kpconv layer
                coord = query_x_low_list[i]  # N_q, 3
                batch = torch.zeros(
                    correlations.shape[0], dtype=torch.int64, device=coord.device
                )
                sigma = 2.0
                radius = 2.5 * self.args.grid_size * sigma
                neighbors = tp.ball_query(
                    radius,
                    self.args.max_num_neighbors,
                    coord,
                    coord,
                    mode="partial_dense",
                    batch_x=batch,
                    batch_y=batch,
                )[
                    0
                ]  # N_q, max_num_neighbors
                correlations = self.kpconv(
                    correlations, coord, batch, neighbors.clone()
                )  # N_q, C

                # classification layer
                out = self.cls(correlations)  # N_q, n_way+1
                sample_predictions.append(out)
                final_pred_sample = (
                pointops.interpolation(
                    query_x_low,
                    query_x[:, :3].cuda().contiguous(),
                    out.contiguous(),
                    query_offset_low,
                    query_offset.cuda(),
                )
                .transpose(0, 1)
                .unsqueeze(0) )  # 1, n_way+1, N_query
                final_pred_sample_list.append(final_pred_sample)  # 1, n_way+1, N_query
                
             # 集成多个采样的结果
            if num_samples == 1:
                final_out = sample_predictions[0]
            else:
                stacked_predictions = torch.stack(sample_predictions, dim=0)
                final_out = stacked_predictions.mean(dim=0)
                final_pred_sample_stack = torch.stack(final_pred_sample_list, dim=0)  # [num_samples, 1, n_way+1, N_query]

                # 计算不确定性
                prob = F.softmax(final_pred_sample_stack, dim=2)  # [num_samples, 1, n_way+1, N_query]
                prob_mean = prob.mean(dim=0).squeeze(0)  # [n_way+1, N_query]
                uncertainty = -(prob_mean * torch.log(prob_mean + 1e-8)).sum(dim=0)  # [N_query]
                all_uncertainties.append(uncertainty) 
            query_pred.append(final_out)
        query_pred = torch.cat(query_pred)  # N_q, n_way+1
        if len(all_uncertainties) > 0:
            all_uncertainties = torch.cat(all_uncertainties)  # N_q
        else:
            all_uncertainties = torch.zeros(query_pred.shape[0], device=query_pred.device)

        # NaN检查
        assert not torch.any(
            torch.isnan(query_pred)
        ), "torch.any(torch.isnan(query_pred))"
        # 损失计算
        loss = self.criterion(query_pred, query_y_low)
        if query_base_y is not None:
            loss += self.criterion_base(q_base_pred, query_base_y.cuda())
        if self.training:
            # 使用warmup的KL权重
            kl_weight = self.get_kl_weight(epoch)
            if len(kl_losses) > 0:
                loss += kl_weight * (sum(kl_losses) / len(kl_losses))
        # 插值恢复原始点数
        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # 1, n_way+1, N_query

        # 可视化
        if self.visualization:
            self.vis(
                query_offset,
                query_x,
                query_y,
                support_offset,
                support_x,
                support_y,
                final_pred,
                all_uncertainties,
            )

        kl_loss_val = sum(kl_losses) / len(kl_losses) if len(kl_losses) > 0 else torch.tensor(0.0)
        return final_pred, loss, 0.0, kl_loss_val

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """
        Get the features of one point cloud from backbone network.

        Args:
            ptclouds: Input point clouds with shape (N_pt, 6), where N_pt is the number of points.
            offset: Offset tensor with shape (b), where b is the number of query scenes.
            gt: Ground truth labels. shape (N_pt).
            query_base_y: Optional base class labels for input point cloud. shape (N_pt).

        Returns:
            feat: Features from backbone with shape (N_down, C), where C is the number of channels.
            coord: Point coords. Shape (N_down, 3).
            offset: Offset for each scene. Shape (b).
            gt: Ground truth labels. Shape (N_down).
            base_pred: Base class predictions from backbone. Shape (N_down, N_base_classes).
            query_base_y: Base class labels for input point cloud. Shape (N_down).
        """
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),
        )
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()
        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]
        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)
        # 主干网络降采样
        feat, coord, offset, gt, base_pred, query_base_y = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )
        feat = self.bk_ffn(feat)
        return feat, coord, offset, gt, base_pred, query_base_y

    def getPrototypes(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            protos = self.getMutiblePrototypes(
                coord_mask, feat_mask, k
            )
            prototypes.append(protos)
        prototypes = torch.cat(prototypes)
        return prototypes

    def query_getPrototypes(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.query_

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        for i in range(0, self.n_way):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            protos = self.getMutiblePrototypes(
                coord_mask, feat_mask, k
            )
            prototypes.append(protos)
        prototypes = torch.cat(prototypes)
        return prototypes
        
    def getMutiblePrototypes(self, coord, feat, num_prototypes):
        """
        Extract k prototypes using furthest point samplling

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.
        Return:
            prototypes: Extracted prototypes. Shape: (num_prototypes, C).
        """
        # when the number of points is less than the number of prototypes, pad the points with zero features
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])
            return feat

        # sample k seeds  by Farthest Point Sampling
        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()
        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes)

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,)

        # aggregating each cluster to form prototype
        prototypes = torch.zeros(
            (num_prototypes, self.feat_dim), device="cuda"
        )
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(
                1
            )  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)
        return prototypes

    def vis(
        self,
        query_offset,
        query_x,
        query_y,
        support_offset,
        support_x,
        support_y,
        final_pred,
        all_uncertainties
    ):
        """
        支持集与查询集可视化（拼为一个大图保存，不使用wandb，用原始类别颜色，支持集增加mask显示，坐标轴比例不压缩）
        alpha: 点云可视化不透明度
        """
        save_dir = os.path.join(
            self.args.vis_save_path,
            "vis_results",
            f"{self.args.target_class}_{self.num_inference_samples}"
        )
        # save_dir = os.path.join(self.args.save_path, "vis_results")
        os.makedirs(save_dir, exist_ok=True)

        rank = getattr(self.args, "rank", 0)

        query_offset_cpu = query_offset[:-1].long().cpu()
        query_x_splits = torch.tensor_split(query_x, query_offset_cpu)
        query_y_splits = torch.tensor_split(query_y, query_offset_cpu)
        vis_pred = torch.tensor_split(final_pred, query_offset_cpu, dim=-1)
        # all_uncertainties 是一个标量列表，需要转换为张量
        all_uncertainties_tensor = torch.tensor(all_uncertainties, device=query_x.device)
        vis_all_uncertainties = torch.tensor_split(all_uncertainties_tensor, query_offset_cpu, dim=-1)

        support_offset_cpu = support_offset[:-1].long().cpu()
        support_x_splits = torch.tensor_split(support_x, support_offset_cpu)
        vis_mask = torch.tensor_split(support_y, support_offset_cpu)
        num_support = len(support_offset_cpu) + 1
        num_query = len(query_offset_cpu) + 1

        total_plots = num_support * 2 + num_query * 4  # 每个query有3个子图
        cols = 3
        rows = (total_plots + cols - 1) // cols

        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        fg_color_red = [1, 0, 0]
        fg_color_blue = [0, 0, 1]
        fg_color = fg_color_blue
        point_size =1
        alpha=1
        # 计算所有点的范围
        all_xyz = []
        for support_x_split in support_x_splits:
            all_xyz.append(support_x_split[:, :3].detach().cpu().numpy())
        for query_x_split in query_x_splits:
            all_xyz.append(query_x_split[:, :3].detach().cpu().numpy())
        all_xyz = np.concatenate(all_xyz, axis=0)
        x_min, y_min, z_min = all_xyz.min(axis=0)
        x_max, y_max, z_max = all_xyz.max(axis=0)
   
        plot_idx = 1
        # 支持集可视化
        for i, support_x_split in enumerate(support_x_splits):
            xyz = support_x_split[:, :3].detach().cpu().numpy()
            color_rgb = support_x_split[:, 3:6].detach().cpu().numpy() * 255.0
            mask_color = vis_mask[i].detach().cpu().numpy()

            # 原始点云
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color_rgb / 255.0, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Support_{i}_Raw', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

            # 支持集mask
            # mask_colors = np.ones((mask_color.shape[0], 3)) * 0.9  # 全部淡灰色
            mask_colors = color_rgb / 255.0  # 默认用原始点云颜色
            mask_colors[mask_color != 0] = fg_color  # 前景设为深蓝色
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=mask_colors, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Support_{i}_Mask', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

        # 查询集可视化
        for i, query_x_split in enumerate(query_x_splits):
            xyz = query_x_split[:, :3].detach().cpu().numpy()
            color_rgb = query_x_split[:, 3:6].detach().cpu().numpy() * 255.0
            gt = query_y_splits[i].detach().cpu().numpy()
            pred = vis_pred[i].squeeze(0).max(0)[1].detach().cpu().numpy()
            uncertainty = vis_all_uncertainties[i].detach().cpu().numpy()
            # 原始点云
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color_rgb / 255.0, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Query_{i}_Raw', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

            # 查询集GT
            # gt_colors = np.ones((gt.shape[0], 3)) * 0.9 # 全部淡灰色
            gt_colors = color_rgb / 255.0  # 背景用原始点云颜色

            gt_colors[gt != 0] = fg_color  # 前景设为深蓝色
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=gt_colors, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Query_GT_{i}', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

            # 查询集预测
            pred_colors = np.ones((pred.shape[0], 3))*0.9
            pred_colors[pred != 0] = fg_color  # 前景设为深蓝色
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=pred_colors, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Query_Pred_{i}', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

            # 不确定性
            # uncertainty = np.clip(uncertainty, a_min=0, a_max=np.percentile(uncertainty, 95))
            uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
            threshold = np.percentile(uncertainty, 90)
            mask = uncertainty >= threshold

            colors = np.ones((uncertainty.shape[0], 3)) * 0.9  # 默认灰色
            if np.any(mask):
                unc_masked = uncertainty[mask]
                unc_masked = (unc_masked - unc_masked.min()) / (unc_masked.max() - unc_masked.min() + 1e-8)
                jet_cmap = matplotlib.cm.get_cmap('jet')
                jet_colors = jet_cmap(unc_masked)[:, :3]
                colors[mask] = jet_colors  # 热力点赋色

            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            ax.view_init(elev=0, azim=0)
            ax.set_proj_type('persp')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, edgecolors='none', alpha=alpha)
            ax.set_title(f'Query_Uncertainty_{i}', pad=20, fontsize=14, loc='center')
            ax.axis('off')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
            plot_idx += 1

        plt.tight_layout()
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        filename = f'episode_vis_{timestamp}_rank{rank}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi = 400)
        plt.close(fig)
