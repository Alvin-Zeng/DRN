import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import ops as ops
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class QueryEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, embed_dim=300, num_layers=1, bidirection=True):
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.biLSTM = nn.LSTM(embed_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)
        self.textualAttention = TextualAttention()
        self.build_extract_textual_command()

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(self.hidden_dim*4, self.hidden_dim)
        for t in range(3):
            qInput_layer2 = ops.Linear(self.hidden_dim, self.hidden_dim*2)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(self.hidden_dim*2, 1)

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations['RELU']
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def forward(self, query_tokens, query_length):

        outputs = []
        query_embedding = self.embedding(query_tokens)
        query_embedding = pack_padded_sequence(query_embedding, query_length, batch_first=True)
        self.biLSTM.flatten_parameters()
        # TODO: h_0, c_0 is zero here
        output, _ = self.biLSTM(query_embedding)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # select the hidden state of the last word individually, since the lengths of query are variable.
        q_vector_list = []
        batch_size = query_length.size(0)
        for i, length in enumerate(query_length):
            h1 = output[i][0]
            hs = output[i][length - 1]
            q_vector = torch.cat((h1, hs), dim=-1)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)

        for cmd_t in range(3):
            outputs.append(self.extract_textual_command(q_vector, output, query_length, cmd_t))
        # output = self.textualAttention(output, q_vector, query_length)

        # Note: the output here is zero-padded, we need slice the non-zero items for the following operations.
        return outputs


class TextualAttention(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(TextualAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_dim = hidden_dim * 2
        self.W1 = nn.Linear(self.hidden_dim, 1)
        self.W2 = nn.Linear(self.q_dim, self.hidden_dim)
        self.W3 = nn.Linear(self.q_dim, self.q_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hidden_sequence, summary_vector, query_length):
        masked_att_map = self._get_attention_map(hidden_sequence, summary_vector, query_length)
        masked_att_map = self.softmax(masked_att_map)
        masked_att_map.data.masked_fill_(masked_att_map.data != masked_att_map.data, 0) # remove nan from softmax on -inf
        query_vector = torch.bmm(masked_att_map.unsqueeze(1), hidden_sequence).squeeze(1)
        return query_vector

    def _get_attention_map(self, hidden_sequence, summary_vector, query_length):
        # hidden_sequence: [batch, max_len, 2 * hidden_dim]
        # summary_vector : [batch, 4 * hidden_dim]
        summary_vector = self.W2(self.relu(self.W3(summary_vector))).unsqueeze(1)
        summary_vector = hidden_sequence * summary_vector
        # [batch, query_max_len]
        att_map = self.W1(summary_vector).squeeze(-1)
        # generate mask
        max_len = max(query_length)
        batch_size = query_length.size(0)
        att_mask = torch.arange(max_len, dtype=query_length.dtype,
                                 device=query_length.device).expand(batch_size, max_len) >= query_length.unsqueeze(1)

        att_map.data.masked_fill_(att_mask.data.byte(), -float('inf'))

        return att_map
