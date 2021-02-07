import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from models import units
import copy


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        x = q
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        context = context + x
        return context, attention

class ScaledDotProductAttention_dim1(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        x = q
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        context = context + x
        return context, attention

class MultiHeadScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len, options):

        super(PositionalEncoding, self).__init__()
        self.options= options

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):


        max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        # input_pos = tensor(pos)
        input_pos = torch.from_numpy(pos).long().to(self.options['device'])
        return self.position_encoding(input_pos), input_pos


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = MultiHeadScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
class MultiHeadAttention_(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention_, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
class EncoderLayer_(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer_, self).__init__()

        self.attention = MultiHeadAttention_(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def padding_mask_sand(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0, 
                 options='options'):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = nn.Linear(vocab_size, model_dim)
        self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, seq_time_step, input_len):
        diagnosis_codes = diagnosis_codes.permute(1, 0, 2)
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        mask = mask.permute(1, 0, 2)
        output = self.pre_embedding(diagnosis_codes)
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        weight = weight * mask - 255 * (1 - mask)
        output = outputs[-1].permute(1, 0, 2)
        weight = weight.permute(1, 0, 2)
        return output, weight


class EncoderNew(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0, 
                 options='options'):
        super(EncoderNew, self).__init__()
        self.options=options

        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
        #      range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.trans_layer = torch.nn.Linear(256*3, 256)

    def multi_fusion(self, x, aux):
        x_control = torch.sigmoid(x)
        aux_var = torch.tanh(aux)
        x_var = x_control * aux_var
        x_final = x_control - x_var
        return x_final

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding        
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        '''
        用 Fusion 机制将embedding 以及 time， position信息放进去
        '''
        # output += time_feature
        # output += output_pos
        output_t = self.multi_fusion(output, time_feature)
        output_p = self.multi_fusion(output, output_pos)
        output = torch.cat((output, output_t, output_p), dim=-1)
        output = self.trans_layer(output)
        '''
        # self_attention_mask = padding_mask(ind_pos, ind_pos)

        # attentions = []
        # outputs = []
        # for encoder in self.encoder_layers:
        #     output, attention = encoder(output, self_attention_mask)
        #     attentions.append(attention)
        #     outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        '''
        return output

class EncoderEval(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0,
                 options='options'):
        super(EncoderEval, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        return output, attention

class EncoderPure(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0, 
                 options='options'):
        super(EncoderPure, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        # self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        # seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2)/180
        # time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        # time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        # output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        return output



# def adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes):
#     batch_time_step = copy.deepcopy(batch_time_step)
#     batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
#     for ind in range(len(batch_diagnosis_codes)):
#         if len(batch_diagnosis_codes[ind]) > max_len:
#             batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
#             batch_time_step[ind] = batch_time_step[ind][-(max_len):]
#         batch_time_step[ind].append(0)
#         batch_diagnosis_codes[ind].append([n_diagnosis_codes - 1])
#     return batch_diagnosis_codes, batch_time_step

class TimeEncoder(nn.Module):
    def __init__(self, batch_size, hidden_size,):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, seq_time_step, final_queries, options, mask):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        # time_weights = self.weight_layer(selection_feature)
        return torch.softmax(selection_feature, 1)


class AuxAttEncoder(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(AuxAttEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.selection_layer = torch.nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.att_func = ScaledDotProductAttention_dim1()

    def forward(self, seq_aux_step, final_queries, options, mask):
        selection_aux = self.tanh(self.selection_layer(seq_aux_step))
        att_output, att_score = self.att_func(selection_aux, final_queries, final_queries)
        selection_feature = torch.sum(att_output, 2, keepdim=True)
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)

        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        # selection_feature = self.relu(self.weight_layer(selection_feature))
        # selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        # selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        # time_weights = self.weight_layer(selection_feature)
        return torch.softmax(selection_feature, 1)


class EncoderAttFusion_6(nn.Module):
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                vocab_size_proc,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(EncoderAttFusion_6, self).__init__()

        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(options['hidden_size'], options['hidden_size'], num_heads, options['hidden_size'], options['hidden_size'], dropout=0.1) for _ in
        #     range(num_layers)])
        self.model_dim = model_dim
        # medical code 的表示
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        vocab_size_all = vocab_size_med+vocab_size_labtest+vocab_size_diag+vocab_size_proc
        bound_ = 1 / math.sqrt(vocab_size_all)
        init.uniform_(self.bias_embedding, -bound_, bound_)
        # labtest code 的表示
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        # diag code 的表示
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        # proc code 的表示
        self.pre_embedding_proc = Embedding(vocab_size_proc, model_dim)

        # medical,labtest,diag,proc code 的 position embedding
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.dropout_emb = nn.Dropout(dp)
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.att_func = ScaledDotProductAttention(0.1)

    # attentiion 模块
    def forward(self, med_codes, labtest_codes, diag_codes, proc_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, proc_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        # 四种/三种 code通过attention机制求和进行融合
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,max_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        output_proc = (self.pre_embedding_proc(proc_codes) * proc_mask_code)
        cat_output = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
        att_output, att_ = self.att_func(cat_output, cat_output, cat_output)
        output_embed = att_output.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding
        output_embed = output_embed + output_pos
        output = self.dropout_emb(output_embed)
        # self_attention_mask = padding_mask(ind_pos, ind_pos)

        # attentions = []
        # outputs = []
        # for encoder in self.encoder_layers:
        #     output, attention = encoder(output, self_attention_mask)
        #     attentions.append(attention)
        #     outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)

        return output

class EncoderAttFusion_5(nn.Module):
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(EncoderAttFusion_5, self).__init__()
        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(options['hidden_size'], options['hidden_size'], num_heads, options['hidden_size'], options['hidden_size'], dropout=0.1) for _ in
        #     range(num_layers)])
        self.model_dim = model_dim
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        vocab_size_all = vocab_size_med+vocab_size_labtest+vocab_size_diag
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_ = 1 / math.sqrt(vocab_size_all)
        init.uniform_(self.bias_embedding, -bound_, bound_)
        # labtest code 的表示
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        # diag code 的表示
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)

        # medical,labtest,diag code 的 position embedding
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.dropout_emb = nn.Dropout(dp)
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.att_func = ScaledDotProductAttention(0.0)

    def multi_fusion(self, x, aux):
        x_control = torch.sigmoid(x)
        aux_var = torch.tanh(aux)
        x_var = x_control * aux_var
        x_final = x_control - x_var
        return x_final

    def forward(self, med_codes, labtest_codes, diag_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        # 四种/三种 code通过attention机制求和进行融合
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,max_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        cat_output = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
        att_output, att_ = self.att_func(cat_output, cat_output, cat_output)
        output_embed = att_output.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding
        output_med = output_embed + output_pos
        output = self.dropout_emb(output_med)
        # self_attention_mask = padding_mask(ind_pos, ind_pos)

        # attentions = []
        # outputs = []
        # for encoder in self.encoder_layers:
        #     output, attention = encoder(output, self_attention_mask)
        #     attentions.append(attention)
        #     outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        
        return output

class Transformer_attentionFusion_Transmain_selfatt_gated(nn.Module):
    def __init__(self, n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, batch_size, options):
        super(Transformer_attentionFusion_Transmain_selfatt_gated, self).__init__()
        # self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size, options['hidden_size'])
        self.aux_encoder = AuxAttEncoder(options['hidden_size'], options['hidden_size'])
        if options['dataset'] == 'mimic_data':
            self.feature_encoder = EncoderAttFusion_6(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, n_proc_codes+1, 31, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        elif options['dataset'] == 'eicu_data':
            self.feature_encoder = EncoderAttFusion_5(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, 11, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        self.self_layer = torch.nn.Linear(options['hidden_size'], 1)
        self.classify_layer = torch.nn.Linear(options['hidden_size'], options['n_labels'])
        self.quiry_layer = torch.nn.Linear(options['hidden_size'], options['hidden_size'])
        self.quiry_weight_layer = torch.nn.Linear(options['hidden_size'], 2)
        self.quiry_2weight_layer = torch.nn.Linear(options['hidden_size'],2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dp']
        self.dropout = nn.Dropout(dropout_rate)
        self.static_trans = nn.Linear(options['static_dim'], options['hidden_size'])
        self.selfatt = ScaledDotProductAttention(0.1)

    def get_self_attention(self, features, query, mask):
        att_output, att_score = self.selfatt(features, features, features)
        attention = torch.softmax(self.self_layer(att_output).masked_fill(mask, -np.inf), dim=1)
        # attention = torch.sum(key * query, 2, keepdim=True) / 8
        return attention

    def forward(self, batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, seq_time_step, batch_labels, options, maxlen, batchS):
        # seq_dignosis_codes: [batch_size, length, bag_len]
        # seq_time_step: [batch_size, length] the day times to the final visit
        # batch_labels: [batch_size] 0 negative 1 positive
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_labels, batch_mask, batch_mask_final, batch_med_mask_code, \
            batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code = units.pad_matrix_new(batch_med_codes, batch_labtest_codes, \
                batch_diag_codes, batch_proc_codes, batch_labels, options)
        batch_med_codes = torch.LongTensor(batch_med_codes).to(options['device'])
        batch_labtest_codes = torch.LongTensor(batch_labtest_codes).to(options['device'])
        batch_diag_codes = torch.LongTensor(batch_diag_codes).to(options['device'])
        batch_mask_mult = torch.BoolTensor(1-batch_mask).unsqueeze(2).to(options['device'])
        batch_mask_final = torch.Tensor(batch_mask_final).unsqueeze(2).to(options['device'])
        batch_med_mask_code = torch.Tensor(batch_med_mask_code).unsqueeze(3).to(options['device'])
        batch_labtest_mask_code = torch.Tensor(batch_labtest_mask_code).unsqueeze(3).to(options['device'])
        batch_diag_mask_code = torch.Tensor(batch_diag_mask_code).unsqueeze(3).to(options['device'])
        batch_s = torch.from_numpy(np.array(batchS)).float().to(options['device'])
        seq_time_step = torch.from_numpy(seq_time_step).float().to(options['device'])
        lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diag_codes])).to(options['device'])
        if options['dataset'] == 'mimic_data':
            batch_proc_mask_code = torch.Tensor(batch_proc_mask_code).unsqueeze(3).to(options['device'])
            batch_proc_codes = torch.LongTensor(batch_proc_codes).to(options['device'])
            features  = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, \
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code, \
                    seq_time_step, lengths, options)
        elif options['dataset'] == 'eicu_data':
            features  = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes, \
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, \
                    seq_time_step, lengths, options)
        
        final_statues = features * batch_mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        # prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        self_weight = self.get_self_attention(features, quiryes, batch_mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, batch_mask_mult)
        attention_weight = torch.softmax(self.quiry_2weight_layer(final_statues), 2)
        trans_main_weight = self.aux_encoder(features,quiryes, options, batch_mask_mult)
        total_weight = torch.cat((trans_main_weight, self_weight), 2)
        total_weight = torch.sum(total_weight*attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        A = weighted_features
        B = features
        weighted_features = A * torch.sigmoid(B)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        static_outputs = self.static_trans(batch_s)
        averaged_features = F.relu(static_outputs+averaged_features)
        predictions = self.classify_layer(averaged_features)
        predictions = torch.sigmoid(predictions)
        labels = torch.LongTensor(batch_labels)
        labels = labels.to(options['device'])
        return predictions, labels, self_weight


## code level 的所有的code 都要与主任务的code有边
class EncoderAllCodeLevelAttFusion_TaskAware_mimic(nn.Module):
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                vocab_size_proc,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(EncoderAllCodeLevelAttFusion_TaskAware_mimic, self).__init__()

        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(options['hidden_size'], options['hidden_size'], num_heads, options['hidden_size'], options['hidden_size'], dropout=0.1) for _ in
        #     range(num_layers)])
        self.model_dim = model_dim
        # medical code 的表示
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding_med = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_med = 1 / math.sqrt(vocab_size_med)
        init.uniform_(self.bias_embedding_med, -bound_med, bound_med)
        # labtest code 的表示
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        self.bias_embedding_labtest = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_labtest = 1 / math.sqrt(vocab_size_labtest)
        init.uniform_(self.bias_embedding_labtest, -bound_labtest, bound_labtest)
        # diag code 的表示
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        self.bias_embedding_diag = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_diag = 1 / math.sqrt(vocab_size_diag)
        init.uniform_(self.bias_embedding_diag, -bound_diag, bound_diag)
        # proc code 的表示
        self.pre_embedding_proc = Embedding(vocab_size_proc, model_dim)
        self.bias_embedding_proc = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_proc = 1 / math.sqrt(vocab_size_proc)
        init.uniform_(self.bias_embedding_proc, -bound_proc, bound_proc)

        # medical,labtest,diag,proc code 的 position embedding
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.att_func_med = ScaledDotProductAttention()
        self.att_func_labtest = ScaledDotProductAttention()
        self.att_func_diag = ScaledDotProductAttention()
        self.att_func_proc = ScaledDotProductAttention()
        self.tanh = nn.Tanh()
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.dropout_emb = nn.Dropout(dp)

    # attentiion 模块
    def forward(self, med_codes, labtest_codes, diag_codes, proc_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, proc_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2)
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        # 四种/三种 code通过attention机制求和进行融合
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,max_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        output_proc = (self.pre_embedding_proc(proc_codes) * proc_mask_code)
        if options['predDiag']:
            merge_output_diag_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_diag_trans = output_diag.view(batch_size*time_step, -1, self.model_dim)
            diag_att, _ = self.att_func_diag(output_diag_trans, merge_output_diag_aux, merge_output_diag_aux)
            output_embed_diag = diag_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_diag
            output_embed_diag = output_embed_diag + output_pos
            output_embed = output_embed_diag

        elif options['predLabtest']:
            merge_output_labtest_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_labtest_trans = output_labtest.view(batch_size*time_step, -1, self.model_dim)
            labtest_att, _ = self.att_func_labtest(output_labtest_trans, merge_output_labtest_aux, merge_output_labtest_aux)
            output_embed_labtest = labtest_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_labtest
            output_embed_labtest = output_embed_labtest + output_pos
            output_embed = output_embed_labtest

        elif options['predProc']:
            merge_output_proc_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_proc_trans = output_proc.view(batch_size*time_step, -1, self.model_dim)
            proc_att, _ = self.att_func_proc(output_proc_trans, merge_output_proc_aux, merge_output_proc_aux)
            output_embed_proc = proc_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_proc
            output_embed_proc = output_embed_proc + output_pos
            output_embed = output_embed_proc

        else:
            merge_output_med_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_med_trans = output_med.view(batch_size*time_step, -1, self.model_dim)
            med_att, _ = self.att_func_med(output_med_trans, merge_output_med_aux, merge_output_med_aux)
            output_embed_med = med_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_med
            output_embed_med = output_embed_med + output_pos
            output_embed = output_embed_med    

        output = self.dropout_emb(output_embed)

        return output

class EncoderAllCodeLevelAttFusion_TaskAware_eicu(nn.Module):
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(EncoderAllCodeLevelAttFusion_TaskAware_eicu, self).__init__()
        # medical code 的表示
        self.model_dim = model_dim
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding_med = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_med = 1 / math.sqrt(vocab_size_med)
        init.uniform_(self.bias_embedding_med, -bound_med, bound_med)
        # labtest code 的表示
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        self.bias_embedding_labtest = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_labtest = 1 / math.sqrt(vocab_size_labtest)
        init.uniform_(self.bias_embedding_labtest, -bound_labtest, bound_labtest)
        # diag code 的表示
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        self.bias_embedding_diag = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_diag = 1 / math.sqrt(vocab_size_diag)
        init.uniform_(self.bias_embedding_diag, -bound_diag, bound_diag)

        # medical,labtest,diag,proc code 的 position embedding
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.att_func_med = ScaledDotProductAttention()
        self.att_func_labtest = ScaledDotProductAttention()
        self.att_func_diag = ScaledDotProductAttention()
        self.tanh = nn.Tanh()        
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.dropout_emb = nn.Dropout(dp)

    # attentiion 模块
    def forward(self, med_codes, labtest_codes, diag_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2)
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        # 四种/三种 code通过attention机制求和进行融合
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,max_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        if options['predDiag']:
            merge_output_diag_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_diag_trans = output_diag.view(batch_size*time_step, -1, self.model_dim)
            diag_att, _ = self.att_func_diag(output_diag_trans, merge_output_diag_aux, merge_output_diag_aux)
            output_embed_diag = diag_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_diag
            output_embed_diag = output_embed_diag + output_pos
            output_embed = output_embed_diag

        elif options['predLabtest']:
            merge_output_labtest_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_labtest_trans = output_labtest.view(batch_size*time_step, -1, self.model_dim)
            labtest_att, _ = self.att_func_labtest(output_labtest_trans, merge_output_labtest_aux, merge_output_labtest_aux)
            output_embed_labtest = labtest_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_labtest
            output_embed_labtest = output_embed_labtest + output_pos
            output_embed = output_embed_labtest

        else:
            merge_output_med_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_med_trans = output_med.view(batch_size*time_step, -1, self.model_dim)
            med_att, _ = self.att_func_med(output_med_trans, merge_output_med_aux, merge_output_med_aux)
            output_embed_med = med_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_med
            output_embed_med = output_embed_med + output_pos
            output_embed = output_embed_med    

        output = self.dropout_emb(output_embed)

        return output

class Transformer_AllCodeLevelattentionFusion_Transmain_selfatt_gated(nn.Module):
    def __init__(self, n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, batch_size, options):
        super(Transformer_AllCodeLevelattentionFusion_Transmain_selfatt_gated, self).__init__()
        # self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(batch_size, options['hidden_size'])
        self.aux_encoder = AuxAttEncoder(options['hidden_size'], options['hidden_size'])
        if options['dataset'] == 'mimic_data':
            self.feature_encoder = EncoderAllCodeLevelAttFusion_TaskAware_mimic(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, n_proc_codes+1, 31, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        elif options['dataset'] == 'eicu_data':
            self.feature_encoder = EncoderAllCodeLevelAttFusion_TaskAware_eicu(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, 11, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        self.self_layer = torch.nn.Linear(options['hidden_size'], 1)
        self.classify_layer = torch.nn.Linear(options['hidden_size'], options['n_labels'])
        self.quiry_layer = torch.nn.Linear(options['hidden_size'], options['hidden_size'])
        self.quiry_weight_layer = torch.nn.Linear(options['hidden_size'], 2)
        self.quiry_2weight_layer = torch.nn.Linear(options['hidden_size'],2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dp']
        self.dropout = nn.Dropout(dropout_rate)
        self.static_trans = nn.Linear(options['static_dim'], options['hidden_size'])
        self.selfatt = ScaledDotProductAttention(0.1)

    def get_self_attention(self, features, query, mask):
        att_output, att_score = self.selfatt(features, features, features)
        attention = torch.softmax(self.self_layer(att_output).masked_fill(mask, -np.inf), dim=1)
        # attention = torch.sum(key * query, 2, keepdim=True) / 8
        return attention

    def forward(self, batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, seq_time_step, batch_labels, options, maxlen, batchS):
        # seq_dignosis_codes: [batch_size, length, bag_len]
        # seq_time_step: [batch_size, length] the day times to the final visit
        # batch_labels: [batch_size] 0 negative 1 positive
        seq_time_step = np.array(list(units.pad_time(seq_time_step, options)))
        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_labels, batch_mask, batch_mask_final, batch_med_mask_code, \
            batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code = units.pad_matrix_new(batch_med_codes, batch_labtest_codes, \
                batch_diag_codes, batch_proc_codes, batch_labels, options)
        batch_med_codes = torch.LongTensor(batch_med_codes).to(options['device'])
        batch_labtest_codes = torch.LongTensor(batch_labtest_codes).to(options['device'])
        batch_diag_codes = torch.LongTensor(batch_diag_codes).to(options['device'])
        batch_mask_mult = torch.BoolTensor(1-batch_mask).unsqueeze(2).to(options['device'])
        batch_mask_final = torch.Tensor(batch_mask_final).unsqueeze(2).to(options['device'])
        batch_med_mask_code = torch.Tensor(batch_med_mask_code).unsqueeze(3).to(options['device'])
        batch_labtest_mask_code = torch.Tensor(batch_labtest_mask_code).unsqueeze(3).to(options['device'])
        batch_diag_mask_code = torch.Tensor(batch_diag_mask_code).unsqueeze(3).to(options['device'])
        batch_s = torch.from_numpy(np.array(batchS)).float().to(options['device'])
        seq_time_step = torch.from_numpy(seq_time_step).float().to(options['device'])
        lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diag_codes])).to(options['device'])
        if options['dataset'] == 'mimic_data':
            batch_proc_mask_code = torch.Tensor(batch_proc_mask_code).unsqueeze(3).to(options['device'])
            batch_proc_codes = torch.LongTensor(batch_proc_codes).to(options['device'])
            features  = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, \
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code, \
                    seq_time_step, lengths, options)
        elif options['dataset'] == 'eicu_data':
            features  = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes, \
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, \
                    seq_time_step, lengths, options)
        
        final_statues = features * batch_mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        # prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        self_weight = self.get_self_attention(features, quiryes, batch_mask_mult)
        # time_weight = self.time_encoder(seq_time_step, quiryes, options, batch_mask_mult)
        attention_weight = torch.softmax(self.quiry_2weight_layer(final_statues), 2)
        trans_main_weight = self.aux_encoder(features,quiryes, options, batch_mask_mult)
        total_weight = torch.cat((trans_main_weight, self_weight), 2)
        total_weight = torch.sum(total_weight*attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        A = weighted_features
        B = features
        weighted_features = A * torch.sigmoid(B)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        static_outputs = self.static_trans(batch_s)
        averaged_features = F.relu(static_outputs+averaged_features)
        predictions = self.classify_layer(averaged_features)
        predictions = torch.sigmoid(predictions)
        labels = torch.Tensor(batch_labels)
        labels = labels.to(options['device'])
        return predictions, labels, self_weight

