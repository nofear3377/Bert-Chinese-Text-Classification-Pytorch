# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
#要初始化的神经网络模型，权重初始化的方法，要排除的层名、不对嵌入层进行初始化，随机种子、用于确保初始化的可重复性
def init_network(model, method='xavier', exclude='embedding', seed=123):
    #遍历模型的所有参数
    for name, w in model.named_parameters():
        #如果参数的名称不包含要排除的层名，则继续执行
        if exclude not in name:
            #如果参数的维度小于2，即不是矩阵形式，则跳过该参数
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            #如果参数是偏置，则将其初始化为常数0
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
#初始化神经网络模型的权重

#训练一个深度学习模型
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    #设置模型为训练模式
    model.train()
    #获取模型的所有参数及其名称
    param_optimizer = list(model.named_parameters())
    #定义不需要衰减的参数类型
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #根据参数是否需要衰减，将参数分组，并设置不同的权重衰减值
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #创建一个BertAdam优化器，用于更新模型的参数
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    #初始化验证集最佳损失为一个很大的数
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 初始化标志位，记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        #在每个epoch中，迭代训练数据
        #enumerate把trains, labels组合成索引序列
        for i, (trains, labels) in enumerate(train_iter):
            #使用模型预测训练数据的输出
            outputs = model(trains)
            #清空模型的梯度
            model.zero_grad()
            #计算交叉熵损失
            loss = F.cross_entropy(outputs, labels)
            #反向传播，计算梯度
            loss.backward()
            #更新模型的参数
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                #将标签数据转移到CPU上
                true = labels.data.cpu()
                #获取模型预测的最大概率类别，并转移到CPU上
                predic = torch.max(outputs.data, 1)[1].cpu()
                #计算训练集的准确率
                train_acc = metrics.accuracy_score(true, predic)
                #在验证集上评估模型的性能
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    #保存模型的状态字典
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test config（配置信息），model（要测试的模型），和test_iter（测试数据迭代器）
    model.load_state_dict(torch.load(config.save_path))
    #加载了之前训练好的模型权重。config.save_path 应该包含模型权重的文件路径，torch.load 用于从该路径加载权重，然后通过 model.load_state_dict 方法将加载的权重应用到 model 对象上
    model.eval()
    #将模型设置为评估模式
    start_time = time.time()
    #测试开始的时间
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    #创建了一个格式化字符串 msg，用于输出测试损失和准确率，并且以特定的格式打印出来。{0:>5.2} 表示第一个参数（即 test_loss）将被格式化为至少宽度为5的字符串，保留两位小数。{1:>6.2%} 表示第二个参数（即 test_acc）将被格式化为至少宽度为6的百分比字符串，保留两位小数
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    #设置模型为评估模式，所有的dropout层会被关闭，所有的batch normalization层会使用整个训练集的均值和方差
    model.eval()
    #初始化了用于累计总损失、所有预测和所有真实标签的变量
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        #创建了一个上下文管理器，确保在其内部的代码块中，PyTorch 不会计算梯度
        for texts, labels in data_iter:
            #遍历 data_iter 中的数据，每次迭代都会得到一批文本数据和对应的标签
            #将当前批次的文本数据输入模型，得到模型的输出
            outputs = model(texts)
            #算了当前批次的损失，使用的是交叉熵损失函数
            loss = F.cross_entropy(outputs, labels)
            #将当前批次的损失加到总损失上
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            #这两行代码将标签和预测结果从 PyTorch 张量转换为 NumPy 数组，并从 GPU（如果可用）移动到 CPU
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    #计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        #生成了分类报告和混淆矩阵，并将它们与准确率、平均损失一起返回
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        #生成分类报告和混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion
    #返回准确率和平均损失
    return acc, loss_total / len(data_iter)
