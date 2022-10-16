from hw_asr.base import BaseModel
import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5,)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5,)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

    def forward(self, x, seq_length):
        '''
        x: BxCxDxT
        seq_length: B, transformed sequence length
        '''
        for conv in self.convs:
            x = conv(x)

            if isinstance(conv, nn.Conv2d):
                mask = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                for i in range(x.shape[0]):
                    l = seq_length[i].item()
                    res = x.shape[-1] - l
                    if res == 0:
                        continue

                    mask[i].narrow(-1, l, res).fill_(1)

                x.masked_fill(mask, 0)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2], -1).permute(2, 0, 1).contiguous()  # TxBxH
        return x


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, batch_norm):
        super(RNNModule, self).__init__()

        self.bidirectional = bidirectional

        self.bn = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True)

    def forward(self, x, output_lengths):
        if self.bn is not None:
            x = self.bn(x.view(x.shape[0] * x.shape[1], -1)).view(x.shape[0], x.shape[1], -1)

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        if self.bidirectional:
            s = x.shape[-1] // 2
            x = x[:, :, :s] + x[:, :, s:]

        return x


'''
Inspired with https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/model.py
'''
class DeepSpeech2(BaseModel):
    def __init__(
        self,
        n_feats,
        n_class,
        n_rnn_layers,
        hidden_size,
        is_bidirectional,
        **batch):

        super().__init__(n_feats, n_class, **batch)

        self.conv = ConvBlock()

        with torch.no_grad():
            dim = self.conv(torch.zeros(1, 1, n_feats, 100).float(), torch.ones(1, dtype=int) * 10).shape[2]

        self.rnns = [RNNModule(dim, hidden_size, is_bidirectional, False)]
        self.rnns.extend([RNNModule(hidden_size, hidden_size, is_bidirectional, True) for i in range(1, n_rnn_layers)])

        self.rnns = nn.ModuleList(self.rnns)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_class, bias=False),
        )

    def forward(self, spectrogram, **batch):
        out_len = self.transform_input_lengths(batch['spectrogram_length'])

        x = self.conv(spectrogram.unsqueeze(1), out_len)

        for rnn in self.rnns:
            x = rnn(x, out_len)

        x = self.fc(x.view(x.shape[0] * x.shape[1], -1)).view(x.shape[0], x.shape[1], -1)
        x = x.transpose(0, 1)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
