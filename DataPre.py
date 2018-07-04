import numpy as np
import h5py
import os
from keras.engine import Model as md
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image as img
from keras_vggface import utils

tn_path = 'train.txt'
tt_path = 'test.txt'

tn_hdf5 = 'train_data.h5'
tt_hdf5 = 'test_data.h5'

def get_feature(model,src):
    n_src = img.img_to_array(src)
    n_src = np.expand_dims(n_src,axis=0)
    n_src = utils.preprocess_input(n_src)
    f = model.predict(n_src)
    return f

def read_data(path,model):
    data = open(path,'r')
    feature = []
    label = []
    for line in data:
        line = line.strip('\n')
        src = img.load_img(line,target_size=(224,224))
        ft = get_feature(model,src)
        lb = line[-9:-7]
        lb = int(lb)
        feature.append(ft)
        label.append(lb)
    return feature,label

def write_hdf5(hdf5file,input_data,output_data):
    h5 = h5py.File(hdf5file,'w')
    h5.create_dataset('feature',data = input_data)
    h5.create_dataset('label',data = output_data)
    h5.close()

vgg_model = VGGFace()
vgg_out = vgg_model.get_layer('fc7').output
vgg_n_model = md(vgg_model.input,vgg_out)

print ('Reading & Create List')
tn_f,tn_l = read_data(tn_path,vgg_n_model)
print ('Write hdf5')
write_hdf5(tn_hdf5,tn_f,tn_l)

print ('Reading & Create List')
tt_f,tt_l = read_data(tt_path,vgg_n_model)
print ('Write hdf5')
write_hdf5(tt_hdf5,tt_f,tt_l)

vgg_n_model.save('vgg.h5')
