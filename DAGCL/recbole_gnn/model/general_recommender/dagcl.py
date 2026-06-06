# -*- coding: utf-8 -*-
r"""
DAGCL: Degree-Adaptive Graph Contrastive Learning for Recommendation
################################################
Based on XSimGCL (Junliang Yu et al., TKDE 2023).
"""

import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender import LightGCN


class DAGCL(LightGCN):
    def __init__(self, config, dataset):
        super(DAGCL, self).__init__(config, dataset)

        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.layer_cl = config['layer_cl']

        # Degree-Adaptive Perturbation: scale eps by node degree
        # eps_u = eps * clamp((deg_u / mean_deg)^alpha, min_scale, max_scale)
        self.degree_adaptive_eps_alpha = float(config['degree_adaptive_eps_alpha'] if 'degree_adaptive_eps_alpha' in config else 0.0)
        self.degree_eps_min_scale = float(config['degree_eps_min_scale'] if 'degree_eps_min_scale' in config else 0.5)
        self.degree_eps_max_scale = float(config['degree_eps_max_scale'] if 'degree_eps_max_scale' in config else 2.0)
        # eps_apply: 'all' (default), 'user_only', 'item_only'
        self.eps_apply = config['eps_apply'] if 'eps_apply' in config else 'all'
        self.register_buffer('_eps_per_node', None)
        if self.degree_adaptive_eps_alpha > 0.0:
            self._build_eps_per_node()

        # CL Loss Reweighting: downweight head users' contrastive loss
        self.cl_head_weight = float(config['cl_head_weight'] if 'cl_head_weight' in config else 1.0)
        self.head_ratio_u = float(config['head_ratio_u'] if 'head_ratio_u' in config else 0.0)
        self.register_buffer('_head_user_mask', None)
        if self.cl_head_weight != 1.0 and self.head_ratio_u > 0.0:
            self._build_head_user_mask()

        # Item-side CL Reweighting: downweight head items' contrastive loss
        self.item_cl_head_weight = float(config['item_cl_head_weight'] if 'item_cl_head_weight' in config else 1.0)
        self.item_head_ratio = float(config['item_head_ratio'] if 'item_head_ratio' in config else 0.0)
        self.register_buffer('_head_item_mask', None)
        if self.item_cl_head_weight != 1.0 and self.item_head_ratio > 0.0:
            self._build_head_item_mask()

    def _node_degree(self):
        n_nodes = self.n_users + self.n_items
        src = self.edge_index[0].cpu()
        deg = torch.bincount(src, minlength=n_nodes).float()
        return deg.to(self.device)

    def _build_eps_per_node(self):
        deg = self._node_degree()
        mean_deg = deg.mean().clamp_min(1.0)
        scale = (deg / mean_deg).pow(self.degree_adaptive_eps_alpha).clamp(
            self.degree_eps_min_scale, self.degree_eps_max_scale
        )
        if self.eps_apply == 'user_only':
            scale[self.n_users:] = 1.0   # items: uniform eps
        elif self.eps_apply == 'item_only':
            scale[:self.n_users] = 1.0   # users: uniform eps
        # 'all': both users and items scaled (default)
        self._eps_per_node = (self.eps * scale).unsqueeze(1)  # (n_nodes, 1)

    def _build_head_user_mask(self):
        user_deg = self._node_degree()[:self.n_users]
        threshold = torch.quantile(user_deg, 1.0 - self.head_ratio_u)
        self._head_user_mask = (user_deg >= threshold)  # (n_users,)

    def _build_head_item_mask(self):
        item_deg = self._node_degree()[self.n_users:]
        threshold = torch.quantile(item_deg, 1.0 - self.item_head_ratio)
        self._head_item_mask = (item_deg >= threshold)  # (n_items,)

    def forward(self, perturbed=False):
        all_embs = self.get_ego_embeddings()
        all_embs_cl = all_embs
        embeddings_list = []

        eps = self._eps_per_node if self._eps_per_node is not None else self.eps
        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * eps
            embeddings_list.append(all_embs)
            if layer_idx == self.layer_cl - 1:
                all_embs_cl = all_embs
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.n_users, self.n_items])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2, weights=None):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        pos_exp = torch.exp(pos_score / self.temperature)
        ttl_exp = torch.exp(ttl_score / self.temperature).sum(dim=1)
        loss = -torch.log(pos_exp / ttl_exp)
        if weights is None:
            return loss.mean()
        weights = weights.to(loss.device, dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-12)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl = self.forward(perturbed=True)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])

        user_cl_w = None
        if self._head_user_mask is not None:
            w = torch.ones(user.shape[0], device=user.device)
            w[self._head_user_mask[user]] = self.cl_head_weight
            user_cl_w = w

        item_cl_w = None
        if self._head_item_mask is not None:
            w = torch.ones(pos_item.shape[0], device=pos_item.device)
            w[self._head_item_mask[pos_item]] = self.item_cl_head_weight
            item_cl_w = w

        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[user], user_all_embeddings_cl[user], user_cl_w)
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[pos_item], item_all_embeddings_cl[pos_item], item_cl_w)

        return mf_loss + self.reg_weight * reg_loss + self.cl_rate * (user_cl_loss + item_cl_loss)
