# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32):
        #初始化空列表，用于储存加载的数据
        contents = []
        #打开文件path进行读取，指定编码为UTF-8
        with open(path, 'r', encoding='UTF-8') as f:
            #使用tqdm库来显示进度条，循环遍历文件中的每一行
            for line in tqdm(f):
                #移除每行末尾的空白字符
                lin = line.strip()
                #如果处理后的行为空，则跳过
                if not lin:
                    continue
                #假设每行数据由内容和标签通过制表符分隔，将行分割成内容和标签
                content, label = lin.split('\t')
                #使用配置中指定的分词器对内容进行分词
                token = config.tokenizer.tokenize(content)
                #在分词结果的开头添加一个特殊的CLS标记
                token = [CLS] + token
                #计算分词后的序列长度
                seq_len = len(token)
                #初始化一个空列表mask，用于存储序列的掩码
                mask = []
                #将分词后的标记转换为对应的ID
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        #如果原始序列长度小于pad_size，则进行填充
                        #创建一个掩码，原始序列部分为1，填充部分为0
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        #对序列进行填充
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        #如果原始序列长度大于或等于pad_size，则截断
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                #将处理后的序列、标签、序列长度和掩码作为一个元组添加到contents列表中
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    #使用配置中的训练数据路径和填充大小加载训练数据
    train = load_dataset(config.train_path, config.pad_size)
    #使用配置中的开发数据路径和填充大小加载开发数据
    dev = load_dataset(config.dev_path, config.pad_size)
    #使用配置中的测试数据路径和填充大小加载测试数据
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    #数据批次列表，批次大小，数据应该被发送到哪个设备上
    def __init__(self, batches, batch_size, device):
        #将传入的batch_size赋值给实例变量
        self.batch_size = batch_size
        self.batches = batches
        #计算可以完整提供的批次数量，即总数据量除以批次大小
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        #检查总数据量是否能被批次大小整除
        if len(batches) % self.n_batches != 0:
            self.residue = True
        #初始化一个索引self.index，用于追踪当前批次的位置
        self.index = 0
        self.device = device

    #将数据转换为张量格式
    def _to_tensor(self, datas):
        #将输入序列转换为长整型张量，并发送到指定的设备上
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #将标签转换为长整型张量，并发送到指定的设备上
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        #将序列长度转换为长整型张量，并发送到指定的设备上
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        #将掩码转换为长整型张量，并发送到指定的设备上
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    #当使用next()函数或在for循环中迭代对象时如何获取下一个元素
    def __next__(self):
        #检查是否有剩余的数据（self.residue为True）
        #检查是否已经到达了最后一个完整的批次（self.index == self.n_batches）
        if self.residue and self.index == self.n_batches:
            #取出剩余的所有数据作为一个批次
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            #更新索引以指向下一个批次，尽管在这种情况下，由于已经没有更多的数据，所以这不会有任何效果
            self.index += 1
            #将取出的数据批次通过_to_tensor方法转换为张量格式
            batches = self._to_tensor(batches)
            return batches

        #如果没有剩余数据，但索引超过了应有的批次数，这意味着所有的数据都已经被迭代完毕
        elif self.index >= self.n_batches:
            #重置索引为0，准备下一次可能的迭代
            self.index = 0
            raise StopIteration
        #正常迭代
        else:
            #根据当前的索引和批次大小取出相应的数据批次
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    #这个方法返回迭代器对象本身，这是迭代器协议的要求
    def __iter__(self):
        #直接返回self，表明DatasetIterater的实例本身就是迭代器
        return self

    #当调用len()函数时如何获取对象的长度。它返回的是数据集可以被分成的批次数量。
    def __len__(self):
        #如果有剩余的数据（self.residue为True），则返回的批次数量需要加一，因为最后一批可能不完整
        if self.residue:
            #在有剩余数据的情况下，返回的批次数量是完整批次的数量加上一个额外的批次
            return self.n_batches + 1
        else:
            #返回完整批次的数量
            return self.n_batches


#根据给定的数据集和配置来构建一个DatasetIterater对象
def build_iterator(dataset, config):
    #传入数据集、批次大小和设备信息
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


#计算从某个开始时间到现在所经过的时间差
def get_time_dif(start_time):
    """获取已使用时间"""
    #获取当前时间作为结束时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
