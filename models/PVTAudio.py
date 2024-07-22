import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from models.attention import SelfAttention

class MultiAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=4, heads=1, fusion='add'):
        """ Class wrapping the MultiAttention part of PGL-SUM; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param None | str pos_enc: The selected positional encoding [absolute, relative].
        :param None | int num_segments: The selected number of segments to split the videos.
        :param int heads: The selected number of global heads.
        :param None | str fusion: The selected type of feature fusion.
        """
        super(MultiAttention, self).__init__()

        # Global Attention, considering differences among all frames
        self.attention = SelfAttention(input_size=input_size, output_size=output_size,
                                       freq=freq, pos_enc=pos_enc, heads=heads)

        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                # Local Attention, considering differences among the same segment with reduce hidden size
                self.local_attention.append(SelfAttention(input_size=input_size, output_size=output_size//num_segments,
                                                          freq=freq, pos_enc=pos_enc, heads=4))
        self.permitted_fusions = ["add", "mult", "avg", "max"]
        self.fusion = fusion
        if self.fusion is not None:
            self.fusion = self.fusion.lower()
            assert self.fusion in self.permitted_fusions, f"Fusion method must be: {*self.permitted_fusions,}"

    def forward(self, x):
        """ Compute the weighted frame features, based on the global and locals (multi-head) attention mechanisms.

        :param torch.Tensor x: Tensor with shape [T, input_size] containing the frame features.
        :return: A tuple of:
            weighted_value: Tensor with shape [T, input_size] containing the weighted frame features.
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        weighted_value, attn_weights = self.attention(x)  # global attention

        if self.num_segments is not None and self.fusion is not None:
            segment_size = math.ceil(x.shape[0] / self.num_segments)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[left_pos:right_pos]
                weighted_local_value, attn_local_weights = self.local_attention[segment](local_x)  # local attentions

                # Normalize the features vectors
                weighted_value[left_pos:right_pos] = F.normalize(weighted_value[left_pos:right_pos].clone(), p=2, dim=1)
                weighted_local_value = F.normalize(weighted_local_value, p=2, dim=1)
                if self.fusion == "add":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                elif self.fusion == "mult":
                    weighted_value[left_pos:right_pos] *= weighted_local_value
                elif self.fusion == "avg":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                    weighted_value[left_pos:right_pos] /= 2
                elif self.fusion == "max":
                    weighted_value[left_pos:right_pos] = torch.max(weighted_value[left_pos:right_pos].clone(),
                                                                   weighted_local_value)

        return weighted_value, attn_weights



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe.transpose(0,1)[:, :x.size(1), :]
        return self.dropout(x)

class PVTAudio(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 audio_size=2048,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVTAudio, self).__init__()
        self.emb_size = emb_size #d_model
        self.audio_size = audio_size
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
        self.audio_linear = nn.Linear(in_features=self.audio_size, out_features=self.emb_size)
        self.linear_1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)
        

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax
        for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        self.linear_2.bias.data.fill_(0.1322)
    
    def forward(self, video_emb, audio_emb):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
        audio_emb = self.audio_linear(audio_emb)
        video_emb = video_emb + audio_emb
        video_emb = self.pos_enc(
            video_emb
        )
        video_emb = self.transformer_encoder(video_emb)
        video_emb = video_emb.contiguous().view(-1, self.emb_size)
        


        # 2-layer NN (Regressor Network)
        y = self.linear_1(video_emb)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        # y = self.sigmoid(y)

        
        # logits = self.fc(video_emb)

        return y.squeeze(), video_emb

class PVTAudioSeq(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 audio_size=2048,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVTAudioSeq, self).__init__()
        self.emb_size = emb_size #d_model
        self.audio_size = audio_size
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
        self.audio_linear = nn.Linear(in_features=self.audio_size, out_features=self.emb_size)
        self.linear_1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)
        

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax
        for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        self.linear_2.bias.data.fill_(0.1322)
    
    def forward(self, video_emb, audio_emb):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
        audio_emb = self.audio_linear(audio_emb).unsqueeze(0)
        audio_emb = self.pos_enc(
          audio_emb
        )
        video_emb = self.pos_enc(
            video_emb
        )
        video_emb = torch.cat((video_emb, audio_emb), axis=1)
        video_emb = self.transformer_encoder(video_emb)

        video_emb = video_emb.contiguous().view(-1, self.emb_size)
        video_emb, audio_emb = video_emb.split(video_emb.shape[0] // 2, dim=0)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(video_emb)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        # y = self.sigmoid(y)

        # 2-layer NN (Regressor Network)
        ya = self.linear_1(audio_emb)
        ya = self.relu(ya)
        ya = self.drop(ya)
        ya = self.norm_linear(ya)

        ya = self.linear_2(ya)
        
        # logits = self.fc(video_emb)

        y = (y + ya) / 2
        # import pdb ; pdb.set_trace()
        return y.squeeze(), video_emb

class PVTAudioSeqPA(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 audio_size=2048,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVTAudioSeqPA, self).__init__()
        self.emb_size = emb_size #d_model
        self.audio_size = audio_size
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
        self.audio_linear = nn.Linear(in_features=self.audio_size, out_features=self.emb_size)
        self.linear_1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)
        

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax
        for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        self.linear_2.bias.data.fill_(0.1322)
    
    def forward(self, video_emb, audio_emb, patarget):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
        audio_emb = self.audio_linear(audio_emb).unsqueeze(0)
        audio_emb = self.pos_enc(
          audio_emb
        )
        video_emb = self.pos_enc(
            video_emb
        )
        video_emb = torch.cat((video_emb, audio_emb), axis=1)
        video_emb = self.transformer_encoder(video_emb)

        video_emb = video_emb.contiguous().view(-1, self.emb_size)
        video_emb, audio_emb = video_emb.split(video_emb.shape[0] // 2, dim=0)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(video_emb)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        # y = self.sigmoid(y)

        # 2-layer NN (Regressor Network)
        ya = self.linear_1(audio_emb)
        ya = self.relu(ya)
        ya = self.drop(ya)
        ya = self.norm_linear(ya)

        ya = self.linear_2(ya)
        
        # logits = self.fc(video_emb)
        y = (y + ya + patarget.unsqueeze(1)) / 3
        # import pdb ; pdb.set_trace()
        return y.squeeze(), video_emb


class PVTAudioSeqSeparateMLPs(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 audio_size=2048,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVTAudioSeqSeparateMLPs, self).__init__()
        self.emb_size = emb_size #d_model
        self.audio_size = audio_size
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
        self.audio_linear = nn.Linear(in_features=self.audio_size, out_features=self.emb_size)
        self.linear_1v = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2v = nn.Linear(in_features=self.linear_1v.out_features, out_features=1)
        self.linear_1a = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2a = nn.Linear(in_features=self.linear_1a.out_features, out_features=1)
        
        self.drop = nn.Dropout(p=0.5)
        self.norm_yv = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_ya = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_linearv = nn.LayerNorm(normalized_shape=self.linear_1v.out_features, eps=1e-6)
        self.norm_lineara = nn.LayerNorm(normalized_shape=self.linear_1a.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax
        for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        self.linear_2v.bias.data.fill_(0.1322)
        self.linear_2a.bias.data.fill_(0.1322)
    
    def forward(self, video_emb, audio_emb):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
        audio_emb = self.audio_linear(audio_emb).unsqueeze(0)
        audio_emb = self.pos_enc(
          audio_emb
        )
        video_emb = self.pos_enc(
            video_emb
        )
        video_emb = torch.cat((video_emb, audio_emb), axis=1)
        video_emb = self.transformer_encoder(video_emb)

        video_emb = video_emb.contiguous().view(-1, self.emb_size)
        video_emb, audio_emb = video_emb.split(video_emb.shape[0] // 2, dim=0)

        # 2-layer NN (Regressor Network)
        y = self.linear_1v(video_emb)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linearv(y)

        y = self.linear_2v(y)
        # y = self.sigmoid(y)

        # 2-layer NN (Regressor Network)
        ya = self.linear_1a(audio_emb)
        ya = self.relu(ya)
        ya = self.drop(ya)
        ya = self.norm_lineara(ya)

        ya = self.linear_2v(ya)
        
        # logits = self.fc(video_emb)

        y = self.sigmoid((y + ya) / 2)

        return y.squeeze(), video_emb

class PVTAudioSeqq(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 audio_size=2048,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVTAudioSeq, self).__init__()
        self.emb_size = emb_size #d_model
        self.audio_size = audio_size
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
        self.audio_linear = nn.Linear(in_features=self.audio_size, out_features=self.emb_size)
        self.linear_1 = nn.Linear(in_features=self.emb_size, out_features=self.emb_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)
        

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=self.emb_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax
        for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        self.linear_2.bias.data.fill_(0.1322)
    
    def forward(self, video_emb, audio_emb):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
        audio_emb = self.audio_linear(audio_emb).unsqueeze(0)
        audio_emb = self.pos_enc(
          audio_emb
        )
        video_emb = self.pos_enc(
            video_emb
        )
        video_emb = torch.cat((video_emb, audio_emb), axis=1)
        video_emb = self.transformer_encoder(video_emb)

        video_emb = video_emb.contiguous().view(-1, self.emb_size)
        video_emb, audio_emb = video_emb.split(video_emb.shape[0] // 2, dim=0)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(video_emb)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        # y = self.sigmoid(y)

        # 2-layer NN (Regressor Network)
        ya = self.linear_1(audio_emb)
        ya = self.relu(ya)
        ya = self.drop(ya)
        ya = self.norm_linear(ya)

        ya = self.linear_2(ya)
        
        # logits = self.fc(video_emb)

        y = (y + ya) / 2
        # import pdb ; pdb.set_trace()
        return y.squeeze(), video_emb

if __name__ == "__main__":
    pass
    # input = torch.rand(1, 100, 1024)
    # model = PVT()
    # out_emb, out_logits = model(input)
