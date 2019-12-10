import numpy as np
import pandas as pd
import imageio as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os, random

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data_location = r'C:\Users\dev368\Documents\Pneumonia\chest-xray-pneumonia\chest_xray'

def encode_data(normal_data, pneum_data, normal_hot, pneum_hot):
    x = normal_data + pneum_data
    y = normal_hot + pneum_hot
    zipped = list(zip(x,y))
    random.shuffle(zipped)
    x,y = zip(*zipped)

    return x,y

def load_data(data_type, nor = 0, pneu = 0):
    normal_data, pneum_data = [], []
    normal_hot, pneum_hot = [], []
    type_location = data_location + '\\' + data_type

    normal_xrays = type_location + '\\NORMAL\\' 
    pneum_xrays = type_location + '\\PNEUMONIA\\'

    normal_files = os.listdir(normal_xrays)
    pneum_files = os.listdir(pneum_xrays)
    
    if nor == 0:
        normal_len = len(normal_files)
    else:
        normal_len = nor

    if pneu == 0:        
        pneum_len = len(pneum_files)
    else:
        pneum_len = pneu

    for x in range(normal_len):
        n_im = io.imread(normal_xrays + normal_files[x])
        n_im = np.array(rgb2gray(n_im)) 
        normal_data.append(n_im)
        normal_hot.append([1,0])

    for x in range(pneum_len):
        p_im = io.imread(pneum_xrays + pneum_files[x])
        p_im = np.array(rgb2gray(p_im))
        pneum_data.append(p_im)
        pneum_hot.append([0,1])
    
    encoded_x, encoded_y = encode_data(normal_data, pneum_data, normal_hot, pneum_hot)

    return encoded_x, encoded_y


