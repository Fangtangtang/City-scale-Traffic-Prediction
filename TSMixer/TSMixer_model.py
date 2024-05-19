import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, encoder_input_size, seq_len, d_model, dropout):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )


        self.channel = nn.Sequential(
            nn.Linear(encoder_input_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, encoder_input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [batch_size, seq_len, feature dimension] 
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(
        self, encoder_input_size, seq_len, pred_len, e_layers=1, d_model=32, dropout=True
    ):
        super(Model, self).__init__()
        # 堆叠层数
        self.layer = e_layers
        self.model = nn.ModuleList(
            [
                ResBlock(encoder_input_size, seq_len, d_model, dropout)
                for _ in range(e_layers)
            ]
        )
        self.pred_len = pred_len
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc):

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        dec_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
