'''
    抽取各个字段，并写入csv文件中
'''

import re
import unicodecsv as ucsv


# 得到每个中药的字典内容
def get_each_medicine_dict():
    each_medicine_dict = dict()
    with open('data/药典2020_new.txt', encoding='utf-8') as f_r:
        index = 0
        for line in f_r.readlines():
            if line.strip():
                # print(line.strip())
                each_medicine_dict.setdefault(index)
                if not each_medicine_dict[index]:
                    each_medicine_dict[index] = []
                each_medicine_dict[index].append(line.strip())
            else:
                index += 1

    print(each_medicine_dict)

    return each_medicine_dict


# 获取中药名、拼音名、英语名
def get_name(contents):
    medicine = contents[0]
    pinyin_name = contents[1]
    english_name = contents[2]
    return medicine, pinyin_name, english_name


# 获取拉丁名、科目、用药部位
def get_lating_kemu_yongyaobuwei(contents):
    kemu = ''; lading = ''; yongyaobuwei = ''
    for item in contents:
        if '本品为' in item and '科' in item:
            kemu = item.split('科')[0].replace('本品为', '') + '科'
            lading = match_something(item)
            if lading:
                tmp_yongyaobuwei = item.split(lading)[-1].split('。')[0].replace('的', '')
                yongyaobuwei = sub_something(tmp_yongyaobuwei)
            return lading, kemu, yongyaobuwei
    return lading, kemu, yongyaobuwei

# 获取含量测定
# 本品含、本品按干燥品计算
def get_hanliangceding(contents):
    hangliangceding = ''
    for item in contents:
        if '本品按干燥品计算' in item:
            hangliangceding = item.split('本品按干燥品计算')[-1].replace('，', '')
            return hangliangceding
        elif '本品含' in item:
            hangliangceding = item.split('本品含')[-1].replace('o', '')
            return hangliangceding
    return hangliangceding


# 获取性味、归经
def get_xingwei_guijing(contents):
    xingwei = ''; guijing = ''
    for item in contents:
        if '【性味与归经】' in item:
            xingwei = item.split('。')[0].replace('【性味与归经】', '')
            guijing = item.split('。')[1]
            return xingwei, guijing
        elif '【性味】' in item:
            xingwei = item.replace('【性味】', '')
            return xingwei, guijing
    return xingwei, guijing


# 获取功能、主治
def get_gongneng_zhuzhi(contents):
    gongneng = ''; zhuzhi = ''
    for item in contents:
        if '【功能与主治】' in item:
            gongneng = item.split('。')[0].replace('【功能与主治】', '')
            zhuzhi = item.split('。')[1]
            return gongneng, zhuzhi
    return gongneng, zhuzhi


# 获取注意
def get_zhuyi(contents):
    zhuyi = ''
    for item in contents:
        if '【注意】' in item:
            zhuyi = item.replace('【注意】', '')
            return zhuyi
    return zhuyi


# 将以上抽取的信息进行合并
def merge():
    each_medicine_dict = get_each_medicine_dict()
    for key, val in each_medicine_dict.items():
        medicine, pinyin_name, english_name = get_name(val)
        lading, kemu, yongyaobuwei = get_lating_kemu_yongyaobuwei(val)
        hanliangceding = get_hanliangceding(val)
        xingwei, guijing = get_xingwei_guijing(val)
        gongneng, zhuzhi = get_gongneng_zhuzhi(val)
        zhuyi = get_zhuyi(val)
        yield [medicine, pinyin_name, english_name, lading, kemu, yongyaobuwei, hanliangceding, xingwei, guijing, gongneng, zhuzhi, zhuyi]


# 主函数
def main():
    with open('data/extract.csv', 'wb') as f_w:
        writer = ucsv.writer(f_w)
        writer.writerow(['中药名', '拼音', '英语名', '拉丁名', '科目', '用药部位', '含量测定', '性味', '归经', '功能', '主治', '注意'])
        for line in merge():
            writer.writerow(line)


# 正则匹配
def match_something(string):
    # pat = re.search('^[A-Za-z. ]+$', string)
    contents = re.findall('[A-Za-z （.）]+', string)
    for content in contents:
        if len(content) > 2:
            return content
    return ''


# 正则替换
def sub_something(string=''):
    # string = '、小黄连刺 Berberis wilsonae Hemsl. 、细叶小槩 Berberis poiretii Schneid.或匙叶小璧 Berberis vernae Schneid. 等同属数种植物干燥根'
    new_str = re.sub('[^\w\u4e00-\u9fff]+', '', string)
    new_str_end = re.sub('[A-Za-z]+', '、', new_str)
    return new_str_end





if __name__ == '__main__':
    main()