# 获取训练、测试、验证数据集


def train_dev_test():
    count = 0
    with open('data/medicine.txt', encoding='utf-8') as f_r:
        for line in f_r.readlines():
            count += 1
            if count < 22000:
                with open('data/train', 'ab') as f_w1:
                    f_w1.write(line.encode())
            elif count < 26000:
                with open('data/dev', 'ab') as f_w2:
                    f_w2.write(line.encode())
            else:
                with open('data/test', 'ab') as f_w3:
                    f_w3.write(line.encode())

train_dev_test()