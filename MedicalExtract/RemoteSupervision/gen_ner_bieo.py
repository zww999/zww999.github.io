import random

random.seed(8)


def load_entity_dict(dict_name="字典库.txt"):
    entity_type = "medicine".split(",")

    with open(dict_name, encoding='utf-8') as file_in:
        # 得到字典库中的每一列组成为列表
        lines = [line.strip() for line in file_in.readlines() if len(line.strip())>1]

    # 随机打乱字典库
    random.shuffle(lines)
    # 得到字典库中词的最大长度
    max_len = max([len(line) for line in lines])
    # 得到类别个数
    length = len(entity_type)
    # 从index为i开始，取所有的i~最后面index 跳步为length（4）的字典词为index=i的种类词
    # entity_type_mapping->{'faculty': ['透空式码头', '加筋土', '加腋', '危岩', '车行道]}
    entity_type_mapping = {entity_type[i]: lines[i::length] for i in range(length)}
    # print(entity_type_mapping)
    entity_type_mapping_reverse = dict()
    for k, v in entity_type_mapping.items():
        for v1 in v:
            entity_type_mapping_reverse[v1] = k
    # entity_type_mapping_reverse->{'透空式码头': 'faculty', '加筋土': 'faculty'}
    # print(entity_type_mapping, entity_type_mapping_reverse, max_len)
    return entity_type_mapping, entity_type_mapping_reverse, max_len


# 将匹配上字典库中的word进行BIE标注
def to_bie(word, ent_type):
    # # 匹配的word如果长度为1直接标为B-type
    # if len(word) == 1:
    #     return f"{word}\tB-{ent_type}\r\n"
    # 匹配的word如果长度为2直接标为B-type  \t E-type
    if len(word) == 2:
        return f"{word[0]} B-{ent_type}\r\n{word[1]} E-{ent_type}\r\n"
    # 匹配的word如果长度为大于2标为B-type  \t I-type \t I-type \t E-type
    else:
        sentence = f"{word[0]} B-{ent_type}\r\n"
        sentence += "\r\n".join([word[x] + f" I-{ent_type}" for x in range(1, len(word) - 1)])
        sentence += "\r\n"
        sentence += f"{word[len(word) - 1]} E-{ent_type}\r\n"

    return sentence


def gen_bieo(text_file="药典2020_new.txt", entity_type_mapping=None,
             entity_type_mapping_reverse=None, max_len=10):
    with open(text_file, encoding='utf-8') as file_in, open('medicine.txt', mode='wb') as fout:
        for _, line in enumerate(file_in):
            # 语料库中的每一行
            line = line.strip()
            if len(line) == 0:
                continue

            start = 0
            while start < len(line):
                found = False
                for pos in range(max_len, 0, -1):
                    # word
                    word = line[start:start + pos]
                    # 如果每一行中的word匹配上字典库中的词则进行BIE标注
                    if word in entity_type_mapping_reverse:
                        fout.write((to_bie(word, entity_type_mapping_reverse[word])).encode('utf-8'))
                        # fout.write("\r\n")
                        found = True
                        break
                # 查找到了字典库的词
                if found:
                    # 再匹配之后的词
                    start += len(word)
                else:
                    # 将没有匹配上的字符标注为'o'
                    if line[start:start + 1].strip() != "":
                        fout.write((line[start:start + 1] + " O").encode('utf-8'))
                        fout.write(("\r\n").encode('utf-8'))
                    start += 1
            fout.write(("end\r\n").encode('utf-8'))


if __name__ == '__main__':
    entity_type_mapping, entity_type_mapping_reverse, max_len = load_entity_dict()
    gen_bieo(entity_type_mapping=entity_type_mapping,
             entity_type_mapping_reverse=entity_type_mapping_reverse,
             max_len=max_len)
