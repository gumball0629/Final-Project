import numpy as np
from PIL import Image as img

ttlst_path = 'test.txt'
tnlst_path = 'train.txt'
ftlst_path = 'FT_test.txt'
tt_path = 'TestData/'
tn_path = 'TrainData/'
ft_path = 'FT_Data/'

def read_data(path):
    data = open(path,'r')
    data_lst = []
    for line in data:
        line = line.strip('\n')
        data_lst.append('/'+line)

    return data_lst

def reshape_pic(lst,path):
    for item in lst:
        i = img.open(item)
        n_i = i.resize((180,240),img.BILINEAR)
        n_i.save(path+item[-10:])

tt = read_data(ttlst_path)
tn = read_data(tnlst_path)
ft = read_data(ftlst_path)

reshape_pic(tt,tt_path)
reshape_pic(tn,tn_path)
reshape_pic(ft,ft_path)
