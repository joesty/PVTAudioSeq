import torch.nn as nn
import torch
import math

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

class PVT(nn.Module):
    def __init__(self, 
                 emb_size=1024,
                 heads=8,
                 enc_layers=4, 
                 dropout=0.1):
        super(PVT, self).__init__()
        self.emb_size = emb_size #d_model
        self.pos_enc = PositionalEncoding(self.emb_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=enc_layers
        )
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
    
    def forward(self, video_emb):
        if len(video_emb.shape) < 3:
            video_emb = video_emb.unsqueeze(0)
        # Add positional embeddings
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

if __name__ == "__main__":
    pass
    # input = torch.rand(1, 100, 1024)
    # model = PVT()
    # out_emb, out_logits = model(input)