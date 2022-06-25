import torch.nn.functional as F
import torch
import dataset


"""
model input:    BatchSize * Seq_len * Embed_dim
                这里对每个曲线，序列长度就是176，Embed维度为1
通过Encoder几层attention输出是什么？同样长度序列？
这里用transformer，不需要padding，因为序列长度都相同，也不需要mask，只是用selfattenion提取特征
位置编码应该要用
"""
# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

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
        return enc_outputs, enc_self_attns


if __name__ == '__main__':

    # net = IOPE_Net(n_channels=1, n_classes=2)
    # print(net)

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net0 = Encode_Net(n_channels=1, n_classes=64)
    net1 = Decode_Net1(n_channels=64, n_classes=1)
    net2 = Decode_Net2(n_channels=64, n_classes=1)

    # 将网络拷贝到deivce中
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)

    # 指定训练集地址，开始训练
    HSI_dataset = dataset.HSI_Loader('../data/all_curve.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=1024,
                                               shuffle=True)
    batch_size = 1024
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(batch_size, 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        break
    print(curve.shape)
    # break
    # 使用网络参数，输出预测结果
    encode_curve = net0(curve)
    a = net1(encode_curve)
    b = net2(encode_curve)
    print(curve[0, 0, :])
    print(f'{a.shape},a:{a[0, 0, :]},\n b:{b[0, 0, :]}')
