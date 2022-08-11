#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import torch
import numpy as np
from importlib import import_module
import utils
import train


if __name__ == '__main__':
    torch.cuda.set_device(1)
    dataset = 'pos'  # 数据集地址
    x = import_module('models.poss_bert')
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    train_iter = utils.bulid_iterator(train_data, config)
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)

    # 模型训练，评估与测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter)
    train.test(config, model, test_iter)
