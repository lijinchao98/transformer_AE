import torch.nn as nn
import torch
from data.dataset import HSI_Loader
from parts import *


"""
model input:    BatchSize * Seq_len * Embed_dim
                这里对每个曲线，序列长度就是176，Embed维度为176
通过Encoder几层attention输出是什么？同样长度序列？
这里用transformer，不需要padding，因为序列长度都相同，也不需要mask，只是用selfattenion提取特征
位置编码应该要用
"""
# Transformer Parameters
d_model = 176  # Embedding Size
d_ff = 704 # FeedForward dimension
d_k = d_v = 22  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 3  # number of heads in Multi-Head Attention


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = Embedding(d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.toAB = ToAB()

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        enc_outputs = self.toAB(enc_outputs)
        return enc_outputs, enc_self_attns

def Decoder(encode_out):

    a = encode_out[:,:,0]
    b = encode_out[:,:,1]
    u = b / (a + b)
    r = (0.084 + 0.170 * u) * u
    r = torch.squeeze(r)

    return r


if __name__ == '__main__':

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = Encoder()

    # 将网络拷贝到deivce中
    net.to(device=device)
    x = torch.rand(32, 176).to(device=device)
    encode_curve = net(x)
    print(f'Encoder结果：{encode_curve[0].size()}')
    print(len(encode_curve[1]), encode_curve[1][0].size())
    out = Decoder(encode_curve[0])
    print(f'Decoder结果：{out.size()}')

