from PIL import Image
import numpy as np
import os
from os.path import join
import pandas as pd
import random
from tqdm import tqdm
import math

def rearrange_pairs_data(x):
    x['pic_twin_2'] = x['twin_2']
    x['twin_2'] = x['twin_1']
    return x

def read_data(path):
    df_data = pd.read_csv(path, names=['twin_1', 'pic_twin_1', 'twin_2', 'pic_twin_2'], skiprows=1, sep='\t')
    df_data = df_data.apply(lambda x: rearrange_pairs_data(x) if math.isnan(x['pic_twin_2']) else x, axis=1)
    df_data['pic_twin_1'] = df_data['pic_twin_1'].astype('int64')
    df_data['pic_twin_2'] = df_data['pic_twin_2'].astype('int64')
    df_data['is_same_person'] = np.where(df_data['twin_1'] == df_data['twin_2'], 1, 0)
    return df_data


def split_train(df_data):

    names = np.unique(np.hstack((df_data['twin_1'].values , df_data['twin_2'].values) ))
    class_balance = 0
    train_val_ratio = 0
    while not (
            (class_balance > 0.48 and class_balance < 0.52) and (train_val_ratio > 0.08 and train_val_ratio < 0.12)) :
        names_for_val = random.choices (names, k=153)  # for more elegant solution: k = random.randint(int(len(names)/10+3), int(len(names)/10+5))
        df_val = df_data[(df_data['twin_1'].isin(names_for_val)) | (df_data['twin_2'].isin(names_for_val))]
        names_df_val = np.unique(np.hstack((df_val['twin_1'].values, df_val['twin_2'].values)))
        df_train = df_data[~((df_data['twin_1'].isin(names_df_val)) | (df_data['twin_2'].isin(names_df_val)))]
        train_val_ratio = len (df_val['is_same_person'])/len (df_data['is_same_person'])

        class_balance = len(df_train[df_train['is_same_person']==1])/len(df_train['is_same_person'])

        list_same_train = np.unique(np.hstack((df_train[df_train['is_same_person']==1]['twin_1'].values , df_train['twin_2'][df_train['is_same_person']==1].values) ))
        list_diff_train = np.unique(np.hstack((df_train[df_train['is_same_person']==0]['twin_1'].values , df_train['twin_2'][df_train['is_same_person']==0].values) ))

    return df_train, df_val


def get_image_pixels ( file_path ) :
    """
        input: jpg file path
        output: jpg normalized pixels
    """
    # load the image
    face_image = Image.open (file_path)
    # produce pixels for face_image
    pixels = np.asarray (face_image)
    # normalize pixels to the range 0-1
    pixels = pixels.astype ('float32')
    pixels /= 255.0

    return pixels

def get_jpg_filename(num):
    len_num = len(num)
    if len_num == 1:
        return '000' + str(num)
    elif len_num == 2:
        return '00' + str(num)
    elif len_num == 3:
        return '0' + str(num)
    else:
        return str(num)

def preprocess_data(jpg_path, df, train_or_test):

    X = [np.zeros((df.shape[0], 250, 250, 1)) for i in range (2)]
    y = np.zeros((df.shape[0], 1))

    for index, row in enumerate(df.values):

            pix1 = get_image_pixels(jpg_path + row[0]+ '/'  + row[0]+ '_' + get_jpg_filename(str(row[1])) + '.jpg')
            X[0][index, :, :, 0] = pix1
            pix2 = get_image_pixels(jpg_path + row[2]+ '/'  + row[2]+ '_' + get_jpg_filename(str(row[3])) + '.jpg')
            X[1][index, :, :, 0] = pix2
            y[index] = row[4]

    list_same_train = np.where(y==1)[0]
    list_diff_train = np.where(y==0)[0]

    print("{train_or_test} data processing completed".format(train_or_test=train_or_test))

    if train_or_test == 'Train':
        return (X, y, list_same_train, list_diff_train)

    return (X, y)

