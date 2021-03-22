import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    """ Standard seq2seq architecture with configurable encoder and decoder. """

    def __init__(self, encoder, decoder, decoder_function= F.log_softmax):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_function = decoder_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variables, 
                input_lengths=None, target_variable=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variables, input_lengths)
        result = self.decoder(inputs= target_variable, 
                                encoder_hidden = encoder_hidden,
                                encoder_outputs = encoder_outputs,
                                function = self.decoder_function,
                                teacher_forcing_ratio = teacher_forcing_ratio
                            )
        
        return result