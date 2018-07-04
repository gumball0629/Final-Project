import numpy as np
from decimal import Decimal
from keras.models import load_model
from keras.preprocessing import image as img
from keras_vggface import utils

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


def pre(path):
    model = load_model(md_path)
    vgg_model = load_model(vgg_path)
    src = img.load_img(path,target_size=(224,224))
    f = get_feature(vgg_model,src)
    f = f.reshape(-1,1,64,64)/255.
    y = model.predict(f)
    tp = top_5(y)
    
    return tp


