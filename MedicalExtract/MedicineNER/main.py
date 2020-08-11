# -*- coding:utf-8 -*-
# 功能描述：模型训练、测试的入口
import pickle
import sys

import yaml

import torch
import torch.optim as optim
from MedicineNER.data_manager import DataManager
from MedicineNER.model import BiLSTMCRF
from MedicineNER.utils import f1_score, get_tags, format_result


class ChineseNER(object):
    
    def __init__(self, entry="train"):
        # 导入训练参数
        self.load_config()
        # 初始化
        self.__init_model(entry)

    def __init_model(self, entry):
        # 模型训练的参数准备
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            # 保存参数
            self.save_params(data)
            # 验证数据集的准备
            dev_manager = DataManager(batch_size=30, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            # 模型的主体使用的是BiLSTM来进行语义编码，CRF用来约束各个标签
            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            # 存储模型参数
            self.restore_model()
        # 模型用来预测的参数准备
        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.restore_model()

    # 模型训练参数保存在config.yml文件中
    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 100,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout":0.5,
                "model_path": "models/",
                "tags": ["medicine"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        # word_embedding的维度大小
        self.embedding_size = config.get("embedding_size")
        # 隐藏层的维度
        self.hidden_size = config.get("hidden_size")
        # 每一个batch导入多少条数据
        self.batch_size = config.get("batch_size")
        # 模型的保存数据
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        # 模型中神经百分之多少激活
        self.dropout = config.get("dropout")
        # 模型一共训练多少轮
        self.epoch = config.get("epoch")

    # 模型在测试过程中进行参数导入
    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    # 训练过程中保存模型的参数
    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        # 使用Adam优化器进行梯度下降算法的优化迭代
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        # 模型一共训练多少轮轮
        for epoch in range(self.epoch):
            index = 0
            # 获取每一个batch的数据
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                # 计算模型训练过程中的损失
                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                    )
                )
                self.evaluate()
                print("-"*50)
                # 梯度回传
                loss.backward()
                # 优化器优化
                optimizer.step()
                # 保存模型
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')

    # 训练过程中的损失计算
    def evaluate(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        print("\teval")
        for tag in self.tags:
            f1_score(labels, paths, tag, self.model.tag_map)

    # 模型训练好之后的预测
    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)
        _, paths = self.model(sentences)

        entities = []
        for tag in self.tags:
            tags = get_tags(paths[0], tag, self.tag_map)
            entities += format_result(tags, input_str, tag)
        return entities

    # 模型对文件中的句子进行实体预测
    def predict_file(self, f_r_path, f_w_path):
        # 去除重复预测的实体
        duplication = set()
        with open(f_r_path, encoding='utf-8') as f_r:
            with open(f_w_path, 'ab') as f_w:
                for line in f_r.readlines():
                    sent = line.split('\t')[-3].strip()
                    res = self.predict(sent)
                    for item in res:
                        entity = item['word']
                        if entity not in duplication:
                            print(entity)
                            duplication.add(entity)
                            f_w.write((entity+'\n').encode())


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("menu:\n\ttrain\n\tpredict")
    #     exit()
    # if sys.argv[1] == "train":
    #     cn = ChineseNER("train")
    #     cn.train()
    # elif sys.argv[1] == "predict":
    #     cn = ChineseNER("predict")
    #     print(cn.predict())

    # 模型训练的入口
    cn = ChineseNER('train')
    cn.train()

    # 模型预测的入口
    # cn = ChineseNER("predict")
    # print(cn.predict())

    # # # 模型对文件中的句子进行实体预测
    # cn = ChineseNER("predict")
    # cn.predict_file('data/medicine.txt', 'data/predict_entity.txt')