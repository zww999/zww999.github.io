'''
    存放功能函数
'''

# 去除空格
def del_space():
    with open('data/药典2020_new.txt', 'wb') as f_w:
        with open('data/药典2020.txt', encoding='utf-8') as f_r:
            for line in f_r.readlines():
                if line.strip():
                    f_w.write(line.encode('utf-8'))