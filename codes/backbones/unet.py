import math
import torch.nn as nn
import torch


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0
    
    def update_model_average(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)
            
    def update_average(self, old, new):
        return old * self.beta + (1 + self.beta) * new
    
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1
        
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super(PositionalEncoding, self).__init__()
        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward(self, t):
        positional_encoding = self.pos_encoding[t].squeeze(1)
        return positional_encoding


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConvBlock, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

        self.gelu = nn.GELU()

    def forward(self, x):
        if self.residual:
            return self.gelu(x + self.layers(x))
        else:
            return self.layers(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(DownSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConvBlock(in_channels, in_channels, residual=True),
            DoubleConvBlock(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t_embedding):
        """
        't_embedding': [batch, time_dim]

        'emb': [batch, time_dim, h, w]
        """
        x = self.layers(x)
        emb = self.emb_layer(t_embedding)
        emb = emb[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(UpSampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels, residual=True),
            DoubleConvBlock(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
        
    def forward(self, x, x_skip, t_embedding):
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels, size, num_heads=4):
        super(SelfAttentionBlock, self).__init__()
        self.num_channels = num_channels
        self.size = size

        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.num_channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 noise_steps=1000,
                 time_dim=256):
        super(UNet, self).__init__()
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        self.inc = DoubleConvBlock(in_channels, 64)
        self.down1 = DownSampleBlock(64, 128)
        self.self_attention1 = SelfAttentionBlock(128, 32)
        self.down2 = DownSampleBlock(128, 256)
        self.self_attention2 = SelfAttentionBlock(256, 16)
        self.down3 = DownSampleBlock(256, 256)
        self.self_attention3 = SelfAttentionBlock(256, 8)

        self.bot1 = DoubleConvBlock(256, 512)
        self.bot2 = DoubleConvBlock(512, 512)
        self.bot3 = DoubleConvBlock(512, 256)

        self.up1 = UpSampleBlock(512, 128)
        self.self_attention4 = SelfAttentionBlock(128, 16)
        self.up2 = UpSampleBlock(256, 64)
        self.self_attention5 = SelfAttentionBlock(64, 32)
        self.up3 = UpSampleBlock(128, 64)
        self.self_attention6 = SelfAttentionBlock(64, 64)
        self.out_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.pos_encoding(t)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.self_attention1(x2)
        x3 = self.down2(x2, t)
        x3 = self.self_attention2(x3)
        x4 = self.down3(x3, t)
        x4 = self.self_attention3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.self_attention4(x)
        x = self.up2(x, x2, t)
        x = self.self_attention5(x)
        x = self.up3(x, x1, t)
        x = self.self_attention6(x)
        output = self.out_layer(x)

        return output
    
class UNet_conditional(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 noise_steps=1000,
                 time_dim=256,
                 num_classes=None):
        super(UNet_conditional, self).__init__()
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        self.inc = DoubleConvBlock(in_channels, 64)
        self.down1 = DownSampleBlock(64, 128)
        self.self_attention1 = SelfAttentionBlock(128, 32)
        self.down2 = DownSampleBlock(128, 256)
        self.self_attention2 = SelfAttentionBlock(256, 16)
        self.down3 = DownSampleBlock(256, 256)
        self.self_attention3 = SelfAttentionBlock(256, 8)

        self.bot1 = DoubleConvBlock(256, 512)
        self.bot2 = DoubleConvBlock(512, 512)
        self.bot3 = DoubleConvBlock(512, 256)

        self.up1 = UpSampleBlock(512, 128)
        self.self_attention4 = SelfAttentionBlock(128, 16)
        self.up2 = UpSampleBlock(256, 64)
        self.self_attention5 = SelfAttentionBlock(64, 32)
        self.up3 = UpSampleBlock(128, 64)
        self.self_attention6 = SelfAttentionBlock(64, 64)
        self.out_layer = nn.Conv2d(64, out_channels, kernel_size=1)
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y):
        t = self.pos_encoding(t)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.self_attention1(x2)
        x3 = self.down2(x2, t)
        x3 = self.self_attention2(x3)
        x4 = self.down3(x3, t)
        x4 = self.self_attention3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.self_attention4(x)
        x = self.up2(x, x2, t)
        x = self.self_attention5(x)
        x = self.up3(x, x1, t)
        x = self.self_attention6(x)
        output = self.out_layer(x)

        return output