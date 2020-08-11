'''
    数据处理部分
'''

from extract import get_each_medicine_dict

def get_all_medicine_name():
    each_medicine_dict = get_each_medicine_dict()
    for key, val in each_medicine_dict.items():
        medicine = val[0].strip()
        print(medicine)

get_all_medicine_name()
