import torch
import torch.nn as nn
from model.fusion import MCB
from model.image_encoder import ImageEncoder
from model.question_encoder import QuestionEncoder
from model.relation_encoder import ActionRelationEncoder, SpatialRelationEncoder, AttributeRelationEncoder
from model.fast_rcnn_classifier import FastRCNNClassifier
from model.explanation_module import ExplanationGenerator


class merge(nn.Module):
    def __init__(self, dataset, i_emb, q_emb, v_relation,
                 joint_embedding, classifier, relation_type, explanation, need_exp):
        super(merge, self).__init__()
        self.name = "merge_%s" % relation_type
        self.relation_type = relation_type
        self.dataset = dataset
        self.i_emb = i_emb
        self.q_emb = q_emb
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.classifier = classifier
        self.explanation = explanation
        self.need_exp = need_exp

    def forward(self, v, b, q, act_pos_emb, spa_adj_matrix,
                att_adj_matrix):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        i_emb = self.i_emb(q)
        q_emb_seq = self.q_emb.forward_all(i_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)

        # [batch_size, num_rois, out_dim]
        if self.relation_type == "action":
            v_emb = self.v_relation.forward(v, act_pos_emb, q_emb_self_att)
        elif self.relation_type == "spatial":
            v_emb = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # attribute
            v_emb = self.v_relation.forward(v, att_adj_matrix,
                                            q_emb_self_att)

        joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)

        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb

        if self.need_exp == 1:
            text = self.explanation.forword(logits)
            return logits, att, text

        return  logits, att


def build_merge(dataset, args):
    classifier = FastRCNNClassifier(args.num_hid, args.num_hid * 2,
                                    dataset.num_ans_candidates, 0.5)
    ie = ImageEncoder(classifier)
    i_emb = ie.forward(dataset.image)
    qe = QuestionEncoder(args.q_num_blocks, args.q_hidden_size, args.q_dropout_rate)
    q_emb = qe.forward(dataset.question)

    if args.relation_type == "semantic":
        v_relation = ActionRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.sem_label_num,
            num_heads=args.num_heads,
            num_steps=args.num_steps, nongt_dim=args.nongt_dim,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias)
    elif args.relation_type == "spatial":
        v_relation = SpatialRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.gauss_kernel_dim, args.embedding_dim)
    else:  # attribute
        v_relation = AttributeRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
            num_heads=args.num_heads, num_steps=args.num_steps,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias)

    joint_embedding = MCB(args.relation_dim, args.num_hid, args.ban_gamma)

    explanation = ExplanationGenerator(args.exp_input_size, args.exp_hidden_size, args.exp_num_layers,
                                       args.exp_num_classes)

    return merge(dataset, i_emb, q_emb, v_relation, joint_embedding,
                 classifier, args.relation_type, explanation, args.need_explanation)
