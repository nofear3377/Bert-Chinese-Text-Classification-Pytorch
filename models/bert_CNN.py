# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        #加载了一个预训练的 BERT 模型，并将其赋值给 self.bert 属性
        self.bert = BertModel.from_pretrained(config.bert_path)
        #遍历了 BERT 模型的所有参数，并将它们的 requires_grad 属性设置为 True，因此训练过程中，这些参数的梯度会被计算和更新
        for param in self.bert.parameters():
            param.requires_grad = True
        #创建一个 nn.ModuleList 对象，其中包含了一系列的二维卷积层。每个卷积层的输入通道数为1，输出通道数为 config.num_filters，卷积核的大小为 (k, config.hidden_size)，其中 k 是 config.filter_sizes 中的一个元素
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        #创建Dropout 层
        self.dropout = nn.Dropout(config.dropout)

        #创建全连接层
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    #卷积层
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        #池化窗口的大小等于输入数据的长度x.size(2)，这样每个通道上的最大值就会被保留下来。.squeeze(2)的作用是去掉输出张量中大小为1的第三维，使得输出数据的维度变成[batch_size, channels]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    #平均池化：x = F.avg_pool1d(x, x.size(2)).squeeze(2)

    #模型的前向传播方法，接受一个参数 x（输入数据）。
    context = x[0]  # 输入的句子
    def forward(self, x):
        #取出句子的内容和掩码
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        #使用 BERT 模型对句子进行编码，同时传入注意力掩码以忽略填充的部分
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out
