# -*- coding: utf-8 -*-
# @Time    : 29/11/23 4:02 PM
# @Author  : Jiaqi Wang
# @Affiliation  : Harbin Institute of Technology Shenzhen
# @Email   : mhwjq1998@gmail.com
# @File    : model_mae.py

import os
# os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
# import timm
# from timm.models.layers import  DropPath
# 注意，SimVTP的Mlp和timm里面的不一样
# from timm.models.vision_transformer import Attention, Mlp
import math
from transformers import MvpModel,MvpTokenizer
# from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
import numpy as np
from Multi_Stream_TransformerEncoder import Multi_Stream_TransformerEncoder,Multi_Stream_TransformerEncoderLayer
def check_nan_inf(input_data,text):
    # 检查是否存在NaN
    nan_check = torch.isnan(input_data)
    if nan_check.any().item():
        print(text,"中输入数据中存在NaN")
        return True

    # 检查是否存在Inf
    inf_check = torch.isinf(input_data)
    if inf_check.any().item():
        print(text,"中输入数据中存在Inf")
        return True
    return False

def Pooler(encoded_embedding,attention_mask):
    return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

def compute_sentencelevel_contrastive_logits(projection_embeddings,inputs_attn_mask_batch,target_input_ids_batch,text_llm):
    batch_size = projection_embeddings.shape[0]
    # target_input_ids_batch = target_input_ids_batch.to(device)
    target_input_ids_batch = target_input_ids_batch
    EEG_features = Pooler(projection_embeddings, inputs_attn_mask_batch)
    # get text feature embedding
    text_attention_mask = torch.clone(inputs_attn_mask_batch)
    # learned temperature parameter
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    # 将每行的最后一个1的位置设为0
    """
    for i in range(text_attention_mask.size(0)):
        # 找到每行的最后一个1的索引
        last_ones_indices = text_attention_mask[i].nonzero()[-1][0]
        # last_one_index = last_ones_indices.item()
        text_attention_mask[i, last_ones_indices] = 0
    """
    # target_input_ids_batch['input_ids'] = target_input_ids_batch['input_ids'].squeeze()
    # target_input_ids_batch['attention_mask'] = target_input_ids_batch['attention_mask'].squeeze()
    Text_features = text_llm(input_ids=target_input_ids_batch['input_ids'], attention_mask=target_input_ids_batch['attention_mask']).last_hidden_state # [N, 768]
    text_attention_mask = target_input_ids_batch['attention_mask']
    Sentence_feature = Pooler(Text_features,text_attention_mask)
    # normalized features
    EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True)  # [N, 768]
    Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True) # [N, 768]
    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_EEG = logit_scale * EEG_features @ Sentence_feature.t()  # [N, N]
    logits_per_text = logit_scale * Sentence_feature @ EEG_features.t()  # [N, N]

    # labels = torch.arange(batch_size, device=device).long()
    labels = torch.arange(batch_size).long()
    total_loss = (F.cross_entropy(logits_per_EEG, labels) +F.cross_entropy(logits_per_text, labels)) / 2
    return total_loss



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # (5000,840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #  (420,) # 840/2
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (5000,1,840)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())  # {Tensor:(16,57,840)}
        # print('[DEBUG] positional embedding size:', self.pe.size())  # {Tensor:(5000,1,840)}
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)

class BandPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(BandPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (5000, 840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (420,) # 840/2
        pe[:, 0::2] = torch.sin(position * div_term)  #
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(pe)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (5000, 1, 840)
        self.register_buffer('pe', pe)

        # Define the coefficients for each segment
        # 0.01, 0.01, 0.8, 0.8, 8, 8, 24, 24
        self.segment_coefficients = nn.Parameter(torch.tensor([2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0]))  # Assuming  8 segments
        # ([0.75, 0.75, 0.9, 0.9, 1.05, 1.05, 1.2, 1.2])
        # ([0.8, 0.8, 0.95, 0.95, 1.1, 1.1, 1.25, 1.25])

    def forward(self, x):
        # self.pe = self.pe.to(x.device)
        # self.segment_coefficients = nn.Parameter(torch.tensor([2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0])).to(x.device)
        # Calculate segment size
        segment_size = x.size(2) // len(self.segment_coefficients)

        # Apply coefficients to each segment
        weighted_pe = self.pe[:, :, :segment_size] * self.segment_coefficients[0]
        for i in range(1, len(self.segment_coefficients)):
            weighted_pe = torch.cat((weighted_pe, self.pe[:, :, i * segment_size:(i + 1) * segment_size] * self.segment_coefficients[i]), dim=2)

        x = x + weighted_pe[:x.size(0), :]
        # return self.dropout(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_e = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_e = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'e':
            x = x + self.drop_path(self.attn(self.norm1_e(x)))
            x = x + self.drop_path(self.mlp(self.norm2_e(x)))
        elif modality == 't':
            x = x + self.drop_path(self.attn(self.norm1_t(x)))
            x = x + self.drop_path(self.mlp(self.norm2_t(x)))
        return x



class CETMAE_project_late(nn.Module):
    """ CET-MAE Model
    """
    # NOTE: embed_dim % num_heads ==0 : 1024%16 768%12
    # NOTE: decoder_embed_dim%decoder_num_heads ==0
    def __init__(self, embed_dim=1024,eeg_dim=840, multi_heads=8, feedforward_dim=2048,trans_layers=6, decoder_embed_dim=840,pretrain_path="./models/huggingface/mvp_multi",
                 norm_layer=nn.LayerNorm, device=0):
        super().__init__()
        print('A CET-MAE Model')
        self.device = torch.device(device)
        self.tokenizer = MvpTokenizer.from_pretrained(pretrain_path)
        # 模态特定的编码
        # 是先将eeg embeddings 映射到1024之后加入模态和位置编码还是在加入后在映射到1024维度
        # 那么其实可以做两个 embed_dim_eeg, embed_dim_text
        # self.text_llm = MvpModel.from_pretrained(pretrain_path)

        self.fc_eeg = nn.Linear(eeg_dim, embed_dim)
        self.act = nn.GELU()

        #  TODO:由于EEG2Text的输入并不需要模态编码，并且EEG2Text是模态到模态之间的转换，所以可以暂时不加，后期可以试一下加入的效果（只在CET-MAE上加，EEG2Text上不加/CET-MAE以及CET-MAE上都加）
        # self.modality_e = nn.Parameter(torch.zeros(1, 1, eeg_dim))
        # self.modality_t = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 对于Audio和Visual的特定编码
        self.pos_embed_e = PositionalEncoding(eeg_dim)
        # self.pos_embed_t = PositionalEncoding(embed_dim)

        # eeg-branch
        # 使用现成的Attention
        """
        使用现成的nn.TransformerEncoder
        """
        self.eeg_encoder_layer = nn.TransformerEncoderLayer(d_model=840, nhead=multi_heads,dim_feedforward=feedforward_dim, batch_first=True,norm_first=False)  # nhead=8
        self.e_branch = nn.TransformerEncoder(self.eeg_encoder_layer, num_layers=trans_layers)  # num_layers=6

        # text-branch
        # use MVP,不需要加入位置编码，MVP里面自己集成好了，
        # NOTE：要考虑输入的是text，而不是一个embeddings，如何加入独特的模态编码？难道还需要进入MVP里面改源码吗？
        self.t_branch = MvpModel.from_pretrained(pretrain_path)
        # self.a = MvpModel.shared()
        # NOTE: 将Text部分冻结
        for param in self.t_branch.parameters():
            param.requires_grad = False


        # unified branch
        # NOTE: 因为Text Embeddings 和 EEG embeddings都存在attention mask，所以选择这种形式
        # NOTE: 这里面的dim_feedforward选用2048是否合适
        # self.unify_encoder_layer = nn.Multi_Stream_TransformerEncoderLayer(d_model=1024, nhead=multi_heads,dim_feedforward=feedforward_dim,batch_first=True,norm_first=False) # nhead=16
        # self.unify_branch = nn.Multi_Stream_TransformerEncoder(self.unify_encoder_layer, num_layers=trans_layers)
        # 非对称
        self.unify_encoder_layer = Multi_Stream_TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True, norm_first=False) # nhead=16
        self.unify_branch = Multi_Stream_TransformerEncoder(self.unify_encoder_layer, num_layers=1)

        # Project to lower dimension for the decoder
        # self.decoder_embed_e = nn.Linear(embed_dim, decoder_embed_dim, bias=True) # eeg branch 1024 => 840

        # token used for masking
        # NOTE: 这个应该是为了填充EEG被掩码部分
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_e = PositionalEncoding(eeg_dim)

        self.eeg_decoder_layers = nn.TransformerEncoderLayer(d_model=840, nhead=multi_heads,dim_feedforward=feedforward_dim, batch_first=True,norm_first=False) # nhead=8
        # self.eeg_decoder = nn.TransformerEncoder(self.eeg_decoder_layers, num_layers=
        # 非对称
        self.eeg_decoder = nn.TransformerEncoder(self.eeg_decoder_layers, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # NOTE : decoder_pred_e： 1024-> 840
        self.decoder_embed_e = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pred_t = nn.Linear(embed_dim, 50267, bias=True)  # decoder, 50267 is the MVP vocab_size

        # NOTE:Add ignore_index
        self.loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        # self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        # print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def initialize_weights(self):

        # NOTE:初始化
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.modality_e, std=.02)
        # torch.nn.init.normal_(self.modality_t, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # not use，but can compare the last position mask
    def eeg_masking_preserve_order(self, x, mask_ratio, attention_mask):
        """
        Perform per-sample masking while preserving the order.
        x: [N, L, D], EEG embeddings
        attention_mask: [N, L], binary mask indicating attention areas
        usage: masked_x, mask, ids_restore, masked_attention_mask = self.eeg_random_masking_unstructured(x, mask_ratio, attention_mask)
        """

        N, L, D = x.shape  # batch, length, dim

        # Calculate the effective length based on attention_mask for each sample
        len_keep = (torch.sum(attention_mask, dim=1) * (1 - mask_ratio)).int()

        # Keep the elements based on effective length without shuffling
        masked_x_list = []
        ids_keep_list = []
        ids_restore_list = []
        max_len = 0
        for i in range(N):
            # 将每一个样本上attention mask 值为1的索引 进行随机排序
            rand_indices = torch.randperm(attention_mask[i].sum().int())
            # 随机生成一个长为len_keep的一个索引序列
            ids_keep_i = torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[:len_keep[i]]]
            ids_remove_i =torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[len_keep[i]:]]
            # 这里拿到ids_keep_i升序
            ids_keep_i_sorted_indices = torch.argsort(ids_keep_i)
            ids_keep_i_sorted = ids_keep_i[ids_keep_i_sorted_indices]

            ids_keep_list.append(ids_keep_i_sorted)

            # 这里拿到ids_remove_i升序
            ids_remove_i_sorted_indices = torch.argsort(ids_remove_i)
            ids_remove_i_sorted = ids_remove_i[ids_remove_i_sorted_indices]

            ids_restore_list.append(ids_remove_i_sorted)
            # 掩码后的
            masked_x_i = torch.index_select(x[i], dim=0, index=ids_keep_i_sorted)
            masked_x_list.append(masked_x_i)
            max_len = max(max_len, len(ids_keep_i_sorted))

        # Pad masked_x_list to have the same length
        masked_x_list = [torch.nn.functional.pad(masked_x_i, (0, 0, 0, max_len - len(masked_x_i))) for masked_x_i in
                         masked_x_list]

        # Stack masked_x_list along the batch dimension
        masked_x = torch.stack(masked_x_list, dim=0)

        # Create masked_attention_mask using ids_restore_tensor
        masked_attention_mask = torch.zeros((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask[i, :len(ids_keep_list[i])] = 1  # Marking positions where attention is applied

        masked_attention_mask_invert = torch.ones((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask_invert[i, :len(ids_keep_list[i])] = 0  # Marking positions where attention is applied

        return masked_x, ids_keep_list, ids_restore_list, masked_attention_mask, masked_attention_mask_invert

    def eeg_masking_preserve_order_last_position(self, x, mask_ratio, attention_mask):
        """
        Perform per-sample masking while ensuring the last position with attention is masked.
        x: [N, L, D], EEG embeddings
        attention_mask: [N, L], binary mask indicating attention areas
        usage: masked_x, mask, ids_restore, masked_attention_mask = self.eeg_random_masking_unstructured(x, mask_ratio, attention_mask)
        """

        N, L, D = x.shape  # batch, length, dim

        # Calculate the effective length based on attention_mask for each sample
        len_keep = (torch.sum(attention_mask, dim=1) * (1 - mask_ratio)).int()

        # Keep the elements based on effective length without shuffling
        masked_x_list = []
        ids_keep_list = []
        ids_restore_list = []
        max_len = 0
        for i in range(N):
            # Find the index of the last position with attention
            # print(attention_mask[i])
            last_attention_index = torch.nonzero(attention_mask[i]).squeeze(1)[-1]

            # Generate random indices excluding the last attention position
            rand_indices = torch.randperm(last_attention_index)
            # 没有被掩码的，即被保留的部分
            ids_keep_i = torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[:len_keep[i]]]


            ids_keep_i_sorted_indices = torch.argsort(ids_keep_i)
            ids_keep_i_sorted = ids_keep_i[ids_keep_i_sorted_indices]
            ids_keep_list.append(ids_keep_i_sorted)

            # Generate indices to restore the original order
            ids_remove_i = torch.nonzero(attention_mask[i]).squeeze(1)[rand_indices[len_keep[i]:]]
            # Ensure the last position with attention is masked
            ids_remove_i = torch.cat((ids_remove_i, torch.tensor([last_attention_index], device=x.device)))

            ids_remove_i_sorted_indices = torch.argsort(ids_remove_i)
            ids_remove_i_sorted = ids_remove_i[ids_remove_i_sorted_indices]
            ids_restore_list.append(ids_remove_i_sorted)

            masked_x_i = torch.index_select(x[i], dim=0, index=ids_keep_i_sorted)
            masked_x_list.append(masked_x_i)
            max_len = max(max_len, len(ids_keep_i_sorted))

        masked_x_list = [torch.nn.functional.pad(masked_x_i, (0, 0, 0, max_len - len(masked_x_i))) for masked_x_i in
                         masked_x_list]

        masked_x = torch.stack(masked_x_list, dim=0)

        # Generate the masked_attention_mask where the last attention position is always masked
        masked_attention_mask = torch.zeros((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask[i, :len(ids_keep_list[i])] = 1

        masked_attention_mask_invert = torch.ones((N, max_len), device=x.device)
        for i in range(N):
            masked_attention_mask_invert[i, :len(ids_keep_list[i])] = 0

        return masked_x, ids_keep_list, ids_restore_list, masked_attention_mask, masked_attention_mask_invert

    def eeg_masking_contain_last_position(self, x, mask_ratio, attention_mask):
        N, L = attention_mask.size()

        masked_attention_mask = torch.zeros_like(attention_mask)
        ids_restore = []

        for i in range(N):

            masked_attention_mask[i] = attention_mask[i]
            masked_indices = []
            # Ensure the last position with attention is masked
            last_attention_index = (attention_mask[i] == 1).nonzero(as_tuple=False).squeeze(-1)[-1]
            masked_attention_mask[i, last_attention_index] = 0
            masked_indices.append(last_attention_index)  # Record masked position

            # Find indices where attention_mask is 1
            mask_indices = torch.nonzero(masked_attention_mask[i]).view(-1)

            # Randomly choose indices to mask based on mask_ratio
            num_mask = int((len(mask_indices)+1) * mask_ratio)-1
            mask_indices_to_change = mask_indices[torch.randperm(len(mask_indices))[:num_mask]]

            # Update masked_attention_mask
            masked_attention_mask[i, mask_indices_to_change] = 0
            masked_indices.extend(mask_indices_to_change)

            ids_restore.append(torch.tensor(masked_indices))

        masked_x = x * masked_attention_mask.unsqueeze(-1)  # Apply mask to EEG embeddings
        masked_attention_mask_invert = torch.logical_not(masked_attention_mask).float()
        # masked_attention_mask_invert = ~masked_attention_mask
        return masked_x, masked_attention_mask, masked_attention_mask_invert, ids_restore

    def eeg_masking_contain_position(self, x, mask_ratio, attention_mask):
        N, L = attention_mask.size()

        masked_attention_mask = torch.zeros_like(attention_mask)
        ids_restore = []

        for i in range(N):

            masked_attention_mask[i] = attention_mask[i]
            masked_indices = []

            # Find indices where attention_mask is 1
            mask_indices = torch.nonzero(masked_attention_mask[i]).view(-1)

            # Randomly choose indices to mask based on mask_ratio
            num_mask = int((len(mask_indices)) * mask_ratio)
            mask_indices_to_change = mask_indices[torch.randperm(len(mask_indices))[:num_mask]]

            # Update masked_attention_mask
            masked_attention_mask[i, mask_indices_to_change] = 0
            masked_indices.extend(mask_indices_to_change)

            ids_restore.append(torch.tensor(masked_indices))

        masked_x = x * masked_attention_mask.unsqueeze(-1)  # Apply mask to EEG embeddings
        masked_attention_mask_invert = torch.logical_not(masked_attention_mask).float()
        # masked_attention_mask_invert = ~masked_attention_mask
        return masked_x, masked_attention_mask, masked_attention_mask_invert, ids_restore

    def mask_batch_text_tokens(
            slef, inputs, tokenizer, mlm_probability=0.15, is_train=True):
        """ modified from transformers.data.data_collator
        Args:
            inputs: (B, L), 2D torch.Tensor, does not work for 1D. It has already been padded.
            tokenizer:
            mlm_probability: float
            is_train: if True use random masking, else mask tokens at fixed position to remove randomness in evaluation.
        """
        # print("Inputs shape:", inputs.shape)
        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        labels_list = labels.tolist()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # print(labels.tolist())
        # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        # special_tokens_mask = []
        # for val in labels_list:
        #     if isinstance(val, int):
        #         print("val 是一个整数:",val)
        #         val = [val]  # 将整数转换为包含单个整数的列表
        #     special_tokens_mask.append(tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True))
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # print(special_tokens_mask)
        # 把special_token_mask的部分置为0.0
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer._pad_token is not None:
            # 生成一个矩阵，每个位置代表是否为padding mask
            padding_mask = labels.eq(tokenizer.pad_token_id)
            padding_mask = padding_mask.to(device=probability_matrix.device)
            #
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 将不是masked_indices的位置label赋值为-100，那么对比学习的位置就是-100的位置
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(
        #     torch.full(labels.shape, 0.5)
        #     ).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(
        #     len(tokenizer), labels.shape,
        #     dtype=torch.long)  # len(tokenizer) == #vocab
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # print(inputs == labels)

        masked_indices = ~torch.eq(labels, tokenizer.pad_token_id)  # 获得非 pad token 的索引
        masked_indices &= (labels != -100)  # 排除被设置为 -100 的索引

        return inputs, labels, masked_indices



    # 这里面输入的都是完整版的eeg embeddings和 text
    def forward_encoder(self, e, e_attn_mask, t, mask_ratio_e, mlm_probability):

        # e = F.normalize(e, p=2, dim=-1)

        e = e+self.pos_embed_e(e)

        # e = e + self.modality_e
        # NOTE: t，一般是tokenized_input['input_ids']，MVP内部自己有位置编码

        # NOTE: 两部分还没有解决： 一是ids_restore_e返回值还不对， 二是如何利用ids_restore_e将长度返回原来大小，而可以计算MSE Loss
        # eeg_masking_last_position  eeg_masking_preserve_order
        # e_masked, ids_keep_list, ids_restore_list, masked_attention_mask, masked_attention_mask_invert = self.eeg_masking_preserve_order_last_position(e, mask_ratio_e, e_attn_mask)

        e_masked, masked_attention_mask, masked_attention_mask_invert, ids_restore = self.eeg_masking_contain_last_position(e, mask_ratio_e, e_attn_mask)
        # ablation study used
        # e_masked, masked_attention_mask, masked_attention_mask_invert, ids_restore = self.eeg_masking_contain_position(e, mask_ratio_e, e_attn_mask)
        # text_mlm_input_ids 用于输入到MVP, text_mlm_labels用于预测，mlm_indices 用于生成 attention mask
        text_mlm_input_ids, text_mlm_labels, mlm_indices= self.mask_batch_text_tokens(t.input_ids, self.tokenizer,mlm_probability=mlm_probability)


        # 获取文本的嵌入向量
        # text_embeddings = self.t_branch.shared(text_mlm_input_ids)

        # 将模态编码应用到文本嵌入向量上
        # text_embeddings_with_modality = text_embeddings + self.modality_t

        # audio and visual stream, independent blocks
        # NOTE:使用Attention block的方式
        # for blk in self.e_branch:
        #     e_branch_embeddings = blk(e_masked,masked_attention_mask)

        # NOTE:transformerencoder的输出为nan，是因为masked_attention_mask出现了全为1的值
        # all_zeros = torch.all(masked_attention_mask_invert == 0, dim=1)
        # rows_with_all_zeros = torch.nonzero(all_zeros).squeeze() # 检查哪些行全为零
        # print("全为零的行索引：", rows_with_all_zeros.tolist()) # 输出全为零的行的索引
        # NOTE:如果不想让transformer的输出为nan，那么不能让masked_attention_mask_invert出现全0行
        # check_nan_inf(e_masked,"e_masked")
        e_branch_embeddings = self.e_branch(e_masked, src_key_padding_mask=masked_attention_mask_invert)
        # e_branch_embeddings = self.e_branch(e_masked)
        # check_nan_inf(e_branch_embeddings, "e_branch_embeddings")
        # e_branch_embeddings = F.relu(self.fc_eeg(e_branch_embeddings)) # 840 -> 1024
        e_branch_embeddings = self.act(self.fc_eeg(e_branch_embeddings))  # 840 -> 1024
        # e_branch_embeddings = F.gelu(self.fc_eeg(e_branch_embeddings))  # 840 -> 1024
        # t_branch_embeddings = self.t_branch( input_ids=text_mlm_input_ids,inputs_embeds=text_embeddings, attention_mask=t.attention_mask).last_hidden_state
        t_branch_embeddings = self.t_branch(input_ids=text_mlm_input_ids, attention_mask=t.attention_mask).last_hidden_state
        unify_embeddings = torch.cat((e_branch_embeddings, t_branch_embeddings), dim=1)

        unify_attention_mask_invert = torch.cat((masked_attention_mask_invert,t.attention_mask_invert),dim=1)

        unify_branch_embeddings = self.unify_branch(unify_embeddings,src_key_padding_mask=unify_attention_mask_invert, modality=None)

        # multi-stream forward process
        # x = self.norm(unify_branch_embeddings)

        _,  L_e,  _ = e_branch_embeddings.shape
        # _ , L_t, _ = x.shape

        x_eeg =  unify_branch_embeddings[:, :L_e, :]
        x_text = unify_branch_embeddings[:, L_e:, :]
        # print("check")
        ce = self.unify_branch(e_branch_embeddings, src_key_padding_mask=masked_attention_mask_invert, modality='e')

        # ce = self.norm_e(ce)

        # for blk in self.blocks_u:
        #     ct = blk(t_branch_embeddings, 't')
        text_attention_mask_invert =t.attention_mask_invert
        text_attention_mask_invert_float = text_attention_mask_invert.to(torch.float32)
        # t_branch_embeddings:(32,58,1024)  text_attention_mask_invert:(32,58)
        # print("check")
        ct = self.unify_branch(t_branch_embeddings, src_key_padding_mask=text_attention_mask_invert_float,  modality='t')

        # x 最后要差分成两部分，一部分是eeg模态，做MAE的mse loss，另一部分直接接入分类头，只计算mask部分的loss cross entropy，ids_restore_e
        return x_eeg, x_text, ids_restore, masked_attention_mask, text_mlm_input_ids, text_mlm_labels, mlm_indices, ce, ct

    def forward_decoder(self,masked_e,  eeg_attn_mask_invert, ids_restore_list):

        """
        将decoder部分拿出来，模拟复原mask部分，其中的self.mask_token部分是可学习向量，
        """
        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        # 在decoder部分，eeg和text就分开做了
        # NOTE:需不需要加入F.relu？
        # e_decoder = F.relu(self.decoder_embed_e(latent_eeg)) # 1024->840
        e_decoder = self.act(self.decoder_embed_e(masked_e))  # 1024->840
        # e_decoder = F.gelu(self.decoder_embed_e(masked_e))  # 1024->840
        # for i in range(latent_eeg.shape[0]):
        #     for j, idx in enumerate(ids_keep_list[i]):
        #         e[i, idx] = e_decoder[i, j]

        # Insert the special token using ids_restore_list
        for i in range(masked_e.shape[0]):
            for idx in ids_restore_list[i]:
                e_decoder[i, idx] = self.mask_token.expand(1, 1, e_decoder.shape[2])  # Ensure self.mask_token has the correct shape

        e = e_decoder + self.decoder_pos_embed_e(e_decoder)

        # only apply for eeg
        # NOTE:考虑把这个不跟换成Transformer Encoder
        # Transformer blocks
        # for blk in self.decoder_blocks:
        #     e = blk(e,eeg_attn_mask)
        e = self.eeg_decoder(e, src_key_padding_mask=eeg_attn_mask_invert)
        e = self.decoder_norm(e)
        check_nan_inf(e, "decoder_eeg")
        return e_decoder, e

    def Pooler(self,encoded_embedding, attention_mask):
        # attention_mask.unsqueeze(-1)  # (32,18,1)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

    def text_Pooler(self,encoded_embedding, attention_mask):
        # attention_mask.unsqueeze(-1)  # (32,18,1)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

    def masked_Pooler(self, encoded_embedding, attention_mask, masked_indices):
        masked_attention_mask = attention_mask.clone()
        masked_attention_mask[masked_indices] = 0  # 将被 mask 的部分设为 0
        sum_embed = (encoded_embedding * masked_attention_mask.unsqueeze(-1)).sum(1)
        sum_mask = masked_attention_mask.sum(-1).unsqueeze(-1)
        pooled_output = sum_embed / sum_mask

        return pooled_output

    def compute_sentencelevel_contrastive_logits(self, eeg_embeddings, eeg_attention, text_embedddings, text_attention, masked_indices):
        batch_size = eeg_embeddings.shape[0] # 32
        # target_input_ids_batch = target_input_ids_batch.to(device)
        EEG_features = self.Pooler(eeg_embeddings, eeg_attention)
        # get text feature embedding
        # text_attention_mask = torch.clone(inputs_attn_mask_batch)
        # learned temperature parameter
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        Sentence_feature = self.masked_Pooler(text_embedddings, text_attention, masked_indices)
        # normalized features
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True)  # [N, 768]
        Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True)  # [N, 768]
        # cosine similarity as logits
        # logit_scale = logit_scale.exp()
        # logits_per_EEG1 = torch.matmul(EEG_features, Sentence_feature.t()) * logit_scale
        logits_per_EEG = logit_scale * EEG_features @ Sentence_feature.t()  # [N, N]
        # logits_per_EEG = EEG_features @ Sentence_feature.t()  # [N, N]
        # logits_per_text1 = torch.matmul(Sentence_feature, EEG_features.t()) * logit_scale
        logits_per_text = logit_scale * Sentence_feature @ EEG_features.t()  # [N, N]
        # logits_per_text = Sentence_feature @ EEG_features.t()  # [N, N]

        # labels = torch.arange(batch_size, device=device).long()
        labels = torch.arange(batch_size).long().to(EEG_features.device)
        total_loss = (F.cross_entropy(logits_per_EEG, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss


    def forward_contrastive(self, eeg_embeddings, text_embeddings, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation

        eeg_embeddings = torch.nn.functional.normalize(eeg_embeddings, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

        total = torch.mm(eeg_embeddings, torch.transpose(text_embeddings, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=eeg_embeddings.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc


    def forward_loss_eeg(self, eeg, pred, eeg_ids_restore_list):
        """
        eeg: [N, L, D], EEG嵌入
        pred: [N, L, D], 预测的EEG嵌入
        eeg_ids_restore_list: 包含每个样本的索引列表
        """
        losses = []

        for i, sample_ids in enumerate(eeg_ids_restore_list):
            # 选取当前样本的对应位置
            eeg_sample = eeg[i][sample_ids]
            pred_sample = pred[i][sample_ids]

            # 计算逐元素的绝对差异和平均损失
            loss_sample = ((pred_sample - eeg_sample) ** 2).mean(dim=-1)

            # 将损失添加到列表中
            losses.append(loss_sample)

        # 将每个样本的损失合并成一个张量
        losses_tensor = torch.cat(losses)

        # 计算掩码嵌入的平均损失
        loss = losses_tensor.mean()

        return loss

    # forward 只要传入 eeg embdeddings 和 text embeddings就好了s
    # NOTE:在之前的代码中使用了，target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100，这里不知道需不需要加入，传入的text_labels还没有用到
    def forward(self, eeg, eeg_attn_mask, eeg_attn_mask_invert, text, mask_ratio_e=0.25, mlm_probability=0.5,mlm_loss_weight=0.5, mae_loss_weight=1.0, contrast_loss_weight=0.01,sim_loss_weight=0.0):

        # 此时的EEG输入为 [Batch,Seq_len,embeddings] eg: (32,58,640)
        # print(eeg_attn_mask[0])
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        # eeg = F.normalize(eeg, p=2, dim=-1)
        latent_eeg, latent_text, eeg_ids_restore_list, masked_attention_mask, text_mlm_inputs_ids, text_mlm_labels, mlm_indices, latent_c_eeg, latent_c_text= self.forward_encoder(eeg, eeg_attn_mask,text, mask_ratio_e, mlm_probability)
        # if check_nan_inf(latent_eeg, "latent_eeg"):
        #     latent_eeg = torch.where(torch.isnan(latent_eeg), torch.tensor(1e-5), latent_eeg).to(latent_eeg.device)
        # if check_nan_inf(latent_text, "latent_text"):
        #     latent_text = torch.where(torch.isnan(latent_text), torch.tensor(1e-5), latent_text).to(latent_text.device)  # 替换为一个很小的数值或者
        project_e, pred_e = self.forward_decoder(latent_eeg, eeg_attn_mask_invert,  eeg_ids_restore_list)
        # check_nan_inf(latent_eeg, "latent_eeg")
        # latent_eeg = torch.tensor(1e-5, requires_grad=True).to(latent_eeg.device)  # 替换为一个很小的数值或者 0

        loss_mae_eeg =self.forward_loss_eeg(eeg,pred_e,eeg_ids_restore_list)
        loss_mae = mae_loss_weight * loss_mae_eeg

        # mlm_logits = F.relu(self.decoder_pred_t(latent_text))
        mlm_logits = self.act(self.decoder_pred_t(latent_text))
        # mlm_logits = F.gelu(self.decoder_pred_t(latent_text))
        loss_mlm = self.loss_mlm(input=mlm_logits.view(-1, 50267), target=text_mlm_labels.view(-1))
        loss_mlm = mlm_loss_weight * loss_mlm


        eeg_embeddings_whole_words = self.Pooler(project_e, masked_attention_mask)
        last_one_indices = (torch.sum(eeg_attn_mask, dim=1).long() - 1).clamp(min=0, max=57) # 减1得到索引位置，避免索引超出范围

        # 通过索引获取对应的 eeg embeddings
        eeg_sentence_embeddings = eeg[torch.arange(eeg.size(0)), last_one_indices]
        cos_sim = torch.nn.functional.cosine_similarity(eeg_embeddings_whole_words, eeg_sentence_embeddings, dim=1)
        loss_sim = 1 - cos_sim.mean()
        loss_sim = sim_loss_weight*loss_sim

        # if contrastive loss is used
        loss_c = self.compute_sentencelevel_contrastive_logits(latent_c_eeg, masked_attention_mask,latent_c_text,text.attention_mask,mlm_indices)
        if torch.isnan(loss_c):
            loss_c = torch.tensor(1e-5, requires_grad=True).to(loss_c.device)  # 替换为一个很小的数值或者 0
        # wo multi-stream
        # loss_c = self.compute_sentencelevel_contrastive_logits(latent_eeg, masked_attention_mask,latent_text,text.attention_mask,mlm_indices)
        # NOTE:这里可以加入mask，这个loss也能迫使模型去预测好mask？
        # loss_c1,c_acc = self.forward_contrastive(latent_c_eeg.mean(dim=1), latent_c_text.mean(dim=1))
        loss_c = contrast_loss_weight * loss_c
        # print(loss_c)

        # loss = loss_mlm+loss_c+loss_sim+loss_mae
        loss = loss_mlm + loss_c + loss_mae
        check_nan_inf(loss, "loss")
        return loss_mae,loss_mlm,loss_c,loss_sim,loss



