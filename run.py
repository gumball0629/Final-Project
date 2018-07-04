import numpy as np
import h5py
from decimal import Decimal
import cv2
from keras.engine import Model 
from keras.models import load_model
from keras.preprocessing import image as img
from keras.optimizers import Adam
#from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import tensorflow as tf

img_path = "FT_test.txt"

md_path = "train.h5"

vgg_path = "vgg.h5"

def get_feature(model,src):
    n_src = img.img_to_array(src)
    n_src = np.expand_dims(n_src,axis=0)
    n_src = utils.preprocess_input(n_src)
    f = model.predict(n_src)
    return f

def top_5(lb):
    tp = np.zeros(shape=(2,5))
    for j in range(0,5):
        for i in range(len(lb[0])):
            if lb[0][i]>tp[1][j]:
                a=0
                for k in range(0,j):
                    if lb[0][i]==tp[1][k]:
                        a=1
                if a==0:
                    tp[1][j] = lb[0][i]
                    tp[0][j] = i+1
    return tp

model = load_model(md_path)
vgg_model = load_model(vgg_path)
lst = open(img_path,'r')
sum=0
for line in lst:
    line = line.strip('\n')
    src = img.load_img(line,target_size=(224,224))
    f = get_feature(vgg_model,src)
    f = f.reshape(-1,1,64,64)/255.
    print('ANS: ',line[-9:-7])
    y = model.predict(f)
    tp = top_5(y)
    for i in range(0,5):
        if tp[0][i] == int(line[-9:-7]):
            sum+=1
        print(Decimal(tp[0][i]).quantize(Decimal('0'))," : ",Decimal(tp[1][i]).quantize(Decimal('0.0000')))
    
    print(sum/50)



