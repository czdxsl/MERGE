import torch
import torch.nn as nn
from torch.autograd import Variable
from model.graph_att import GAttNet as GAT
from model.fc import FCNet
from model.gauss_kernal import SpatialGraphClassifier as SGC


def q_expand_v_cat(q, v, mask=True):
    q = q.view(q.size(0), 1, q.size(1))
    repeat_vals = (-1, v.shape[1], -1)
    q_expand = q.expand(*repeat_vals)
    if mask:
        v_sum = v.sum(-1)
        mask_index = torch.nonzero(v_sum == 0)
        if mask_index.dim() > 1:
            q_expand[mask_index[:, 0], mask_index[:, 1]] = 0
    v_cat_q = torch.cat((v, q_expand), dim=-1)
    return v_cat_q


class ActionRelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, pos_emb_dim,
                 nongt_dim, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        super(ActionRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        print("In ActionRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
        in_dim = out_dim + q_dim
        self.action_relation = GAT(dir_num, 1, in_dim, out_dim,
                                   nongt_dim=nongt_dim,
                                   label_bias=label_bias,
                                   num_heads=num_heads,
                                   pos_emb_dim=pos_emb_dim)

    def forward(self, v, position_embedding, q):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q: [batch_size, q_dim]
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]

        Returns:
            output: [batch_size, num_rois, out_dim,3]
        """
        # [batch_size, num_rois, num_rois, 1]
        act_adj_mat = Variable(
            torch.ones(
                v.size(0), v.size(1), v.size(1), 1)).to(v.device)
        act_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            v_cat_q = q_expand_v_cat(q, act_v, mask=True)
            act_v_rel = self.action_relation.forward(v_cat_q,
                                                     act_adj_mat,
                                                     position_embedding)
            if self.residual_connection:
                act_v += act_v_rel
            else:
                act_v = act_v_rel
        return act_v


class SpatialRelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, gauss_kernel_dim, embedding_dim):
        super(SpatialRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        print("In SpatialRelationEncoder, num of graph propogation steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
        in_dim = out_dim + q_dim
        self.spatial_relation = SGC(in_dim, out_dim, gauss_kernel_dim, embedding_dim)

    def forward(self, v, spa_adj_matrix, q):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q: [batch_size, q_dim]
            spa_adj_matrix: [batch_size, num_rois, num_rois, num_labels]

        Returns:
            output: [batch_size, num_rois, out_dim]
        """
        spa_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            v_cat_q = q_expand_v_cat(q, spa_v, mask=True)
            spa_v_rel = self.spatial_relation.forward(v_cat_q, spa_adj_matrix)
            if self.residual_connection:
                spa_v += spa_v_rel
            else:
                spa_v = spa_v_rel
        return spa_v


class AttributeRelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, pos_emb_dim,
                 nongt_dim, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        super(AttributeRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        print("In AttributeRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
        in_dim = out_dim + q_dim
        self.attribute_relation = GAT(dir_num, 1, in_dim, out_dim,
                                      nongt_dim=nongt_dim,
                                      label_bias=label_bias,
                                      num_heads=num_heads,
                                      pos_emb_dim=pos_emb_dim)

    def forward(self, v, position_embedding, q):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q: [batch_size, q_dim]
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]

        Returns:
            output: [batch_size, num_rois, out_dim,3]
        """
        # [batch_size, num_rois, num_rois, 1]
        att_adj_mat = Variable(
            torch.ones(
                v.size(0), v.size(1), v.size(1), 1)).to(v.device)
        att_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            v_cat_q = q_expand_v_cat(q, att_v, mask=True)
            att_v_rel = self.attribute_relation.forward(v_cat_q,
                                                        att_adj_mat,
                                                        position_embedding)
            if self.residual_connection:
                att_v += att_v_rel
            else:
                att_v = att_v_rel
        return att_v
