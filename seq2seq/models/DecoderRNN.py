import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DecoderRNN:
    """ Provides functionality for decoding in Seq2Seq framework, with option for attention """
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id,
                    n_layers = 1, rnn_cell='gru', bidirectional=False, 
                    input_dropout_p = 0, dropout_p= 0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, 
                    hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.rnn_cell = GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout = dropout_p)

        self.output_size = vocab_size
        self.max_length  = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(0)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[]
