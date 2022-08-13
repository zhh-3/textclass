# -*- coding: UTF-8 -*-

import os
import logging

import torch
import transformers
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM
import time
from datetime import timedelta

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
transformers.set_seed(1)
logging.basicConfig(level=logging.INFO)

class LecCallTag():

    # 原始样本统计
    def data_show(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        logging.info("获取数据：%s" % len(data))
        tags_data_dict = {}
        for line in data:
            text_label = line.strip("\n").split('\t')
            if text_label[1] in tags_data_dict:
                tags_data_dict[text_label[1]].append(text_label[0])
            else:
                tags_data_dict[text_label[1]] = [text_label[0]]
        logging.info("其中，各分类数量：")
        for k, v in tags_data_dict.items():
            logging.info("%s: %s" % (k, len(v)))
        return tags_data_dict

    # 数据处理
    def data_process(self, data_file, labels):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [line.strip("\n").split('\t') for line in f.readlines()]  # 电潜泵电缆悬空未固定，大风天气可能导致电缆晃动绝缘磨损	非此类无无无
        text = ["可能性是[MASK][MASK][MASK][MASK][MASK][MASK]的问题有" + _[0] for _ in data]
        label = ['可能性是' + _[1] + '的问题有' + _[0] for _ in data]
        return text, label

    # model, tokenizer
    def create_model_tokenizer(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        return tokenizer, model

    # 构建dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        X_train, X_test, Y_train, Y_test = train_test_split(text, label, test_size=0.1, random_state=1)
        logging.info('训练集：%s条，\n测试集：%s条' % (len(X_train), len(X_test)))
        train_dict = {'text': X_train, 'label_text': Y_train}
        test_dict = {'text': X_test, 'label_text': Y_test}
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding=True, truncation=True, max_length=max_len)
            text_token['labels'] = np.array(
                tokenizer(examples['label_text'], padding=True, truncation=True, max_length=max_len)[
                    "input_ids"])  # 注意数据类型
            return text_token

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, test_dataset

    # 构建trainer
    def create_trainer(self, model, train_dataset, test_dataset, checkpoint_dir, batch_size):
        args = TrainingArguments(
            checkpoint_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            labels = pred.label_ids[:, 10]
            preds = pred.predictions[:, 10].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        return trainer


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def main():
    labels = ["后果发生频率", "非此类无无无"]
    lct = LecCallTag()
    data_file = 'data/prompt.txt'
    checkpoint_dir = "data/saved_dict/"
    save_dir = "prompt_pretrain/"
    batch_size = 32
    max_len = 32
    text, label = lct.data_process(data_file, labels)
    tokenizer, model = lct.create_model_tokenizer("bert-base-chinese")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, test_dataset, checkpoint_dir, batch_size)
    trainer.train()
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


if __name__ == '__main__':
    start_time = time.time()
    main()
    time_dif = get_time_dif(start_time)
    print("总时间：", time_dif)
