# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

#建一个ArgumentParser对象，用于解析命令行参数，description参数提供了对这个脚本功能的简单描述
parser = argparse.ArgumentParser(description='Chinese Text Classification')
#添加一个命令行参数--model，这个参数是必需的，类型为字符串，help参数提供了对这个参数的说明
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert')
#解析命令行参数，并将结果存储在args对象中
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert/bert-cnn/bert-rnn
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    #设置NumPy的随机种子为1，以保证随机操作的可重复性
    np.random.seed(1)
    #设置PyTorch的CPU随机种子为1
    torch.manual_seed(1)
    #设置PyTorch的CUDA随机种子为1
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    #数据迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

#用于设置训练环境、加载数据、构建模型，并启动训练过程
