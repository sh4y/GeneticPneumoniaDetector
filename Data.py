import numpy as np
import pandas as pd
import imageio as io
from skimage.color import rgb2gray
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os, random

data_location = r'C:\Users\dev368\Documents\Pneumonia\chest-xray-pneumonia\chest_xray'

def encode_data(normal_data, pneum_data, normal_hot, pneum_hot):
    x = normal_data + pneum_data
    y = normal_hot + pneum_hot
    zipped = list(zip(x,y))
    random.shuffle(zipped)
    x,y = zip(*zipped)
    return np.asarray(x), np.asarray(list(y))

    
def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def get_max_image_size(dataset):
    max_rows, max_cols = 0, 0
    for image in dataset:
        max_rows = max(image.shape[0], max_rows)
        max_cols = max(image.shape[1], max_cols)
    return (max_rows, max_cols)

def pad_images(images, given_shape=(0,0)):
    max_rows, max_cols = 0, 0
    if (given_shape == (0,0)):
        max_rows, max_cols = get_max_image_size(images)
    else:
        max_rows = given_shape[0]
        max_cols = given_shape[1]
    print("Greatest image dimension is ({0},{1})".format(max_rows,max_cols))
    padded_images = []
    for image in images:
        rows = image.shape[0]
        cols = image.shape[1]
        
        rows_to_pad = max_rows - rows
        cols_to_pad = max_cols - cols

        padded_image = pad(image, (max_rows, max_cols), [rows_to_pad, cols_to_pad])

        padded_images.append(padded_image)
        '''pad_left = cols_to_pad // 2
        pad_right = cols_to_pad - pad_left

        pad_top = rows_to_pad // 2
        pad_bot = rows_to_pad - pad_bot'''
    print("Padded all images to maximum size.")
    return np.asarray(padded_images)

def reshape_training_data(train_dataset):
    nsamples, nx, ny = train_dataset.shape
    d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
    return d2_train_dataset

def load_data(data_type, nor = 0, pneu = 0, pad_shape=(0,0)):
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
    padded_x = pad_images(encoded_x)
    reshaped_x = reshape_training_data(padded_x)    
    return reshaped_x, encoded_y

        


