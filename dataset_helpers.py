from PIL import Image
import numpy as np
import os
from os.path import join
import pandas as pd
import random


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


def preprocess_data(directory, df, train_or_test):
    """ input: directory-jpg file directory
               txt_files_path-train and test txt files path

    output:
    X and Y are np arrays.
    X[0] holds person a pixles X[1] holds person b pixles Y holds the lable.
    All match the index (represents pair id)

    """
    #convert float numbers of the 4th column (exist to no match pairs) to int and all nan to '-1'
    # path_txt = join(txt_files_path, file_name)
    # df = pd.read_csv(path_txt, names=['col1', 'col2', 'col3', 'col4'], skiprows=1, sep='\t')
    # df['col4'].fillna(-1, inplace=True)
    # df['col4'] = df['col4'].astype('int64')
    pairs = df.shape[0]
    X = [np.zeros((pairs, 250, 250, 1)) for i in range (2)]
    Y = np.zeros((pairs, 1))

    for index, row in df.iterrows():
        if str(row['col4']) == '-1':
            pix1 = join(directory, os.fsencode(row['col1']))
            pix1_num = row['col1'] + '_' + get_jpg_filename(str(row['col2'])) + '.jpg'
            pix1_num = os.fsencode(pix1_num)
            pix1 = join(pix1, pix1_num)
            pix1 = get_image_pixels(pix1)
            X[0][index, :, :, 0] = pix1

            pix2 = join(directory, os.fsencode (row['col1']))
            pix2_num = row['col1'] + '_' + get_jpg_filename(str(row['col3'])) + '.jpg'
            pix2_num = os.fsencode(pix2_num)
            pix2 = join(pix2, pix2_num)
            pix2 = get_image_pixels(pix2)
            X[1][index, :, :, 0] = pix2

            Y[index] = 1
        else:
            pix1 = join(directory, os.fsencode(row['col1']))
            pix1_num = row['col1'] + '_' + get_jpg_filename(str (row['col2'])) + '.jpg'
            pix1_num = os.fsencode(pix1_num)
            pix1 = join(pix1, pix1_num)
            pix1 = get_image_pixels(pix1)
            X[0][index, :, :, 0] = pix1

            pix2 = join(directory, os.fsencode(row['col3']))
            pix2_num = row['col3'] + '_' + get_jpg_filename (str(row['col4'])) + '.jpg'
            pix2_num = os.fsencode(pix2_num)
            pix2 = join(pix2, pix2_num)
            pix2 = get_image_pixels(pix2)
            X[1][index, :, :, 0] = pix2

            Y[index] = 0
    print("{train_or_test} data processing completed".format(train_or_test = train_or_test) )

    return (X,Y)

def list_names_df(txt_files_path, file_name):
    path_txt = join(txt_files_path, file_name)
    df = pd.read_csv(path_txt, names=['col1', 'col2', 'col3', 'col4'], skiprows=1, sep='\t')
    df['col4'].fillna(-1, inplace=True)
    df['col4'] = df['col4'].astype ('int64')
    return df

def find_paired(parent_index, name, set_names, df):
    for row,index in df.iterrows():
        if (index>parent_index):
            if (row['col1'] == name):
                set_names.add(['col3'])
            elif (row['col3'] == name):
                set_names.add(row['col1'])
    return index, set_names

def split_train(txt_files_path, file_name):
    df = list_names_df(txt_files_path, file_name)
    val_num = round(0.2*len(df))
    train_num = round(len(df)-val_num)
    set_val_names = set()
    set_train_names = set()
    set_names = set()
    for index, row in df.iterrows():
        rand = random.random()
        if train_num <= 0:
            continue
        elif val_num <= 0:
            continue
        else:
            if (rand <= 0.2):
                if row['col1'] not in set_val_names or row['col1'] not in set_train_names:
                    set_names.add(row['col1'])
                    if row['col4'] != -1 :
                        index_sub = 0
                        while (index_sub < 2199) :
                            index_sub, set_names = find_paired (index, row['col1'], set_names, df)
                            val_num -= len(set_names)
                            set_val_names.update(set_names)
                    else :
                        val_num -= len(set_names)
                        set_val_names.update(set_names)
            else:
                if row['col1'] not in set_val_names or row['col1'] not in set_train_names:
                    set_names.add(row['col1'])
                    if row['col4'] != -1:
                        index_sub = 0
                        while (index_sub < 2199):
                            index_sub, set_names = find_paired(index, row['col1'], set_names, df)
                            train_num -= len(set_names)
                            set_train_names.update(set_names)
                    else :
                        train_num -= len(set_names)
                        set_train_names.update(set_names)
    df_train = pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4'])
    df_val = pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4'])
    for index, row in df.iterrows():
        if row['col4'] == -1:
            if row['col1'] in set_val_names:
                df_val = df_val.append({'col1': row['col1'], 'col2': row['col2'], 'col3': row['col3'], 'col4': row['col4']}, ignore_index=True)
            else:
                df_train = df_train.append({'col1': row['col1'], 'col2': row['col2'], 'col3': row['col3'], 'col4': row['col4']}, ignore_index=True)
        else:
            if row['col1'] in set_val_names or row['col3'] in set_val_names:
                df_val = df_val.append ({'col1' : row['col1'], 'col2' : row['col2'], 'col3' : row['col3'], 'col4' : row['col4']},ignore_index=True)
            else:
                df_train = df_train.append({'col1': row['col1'], 'col2': row['col2'], 'col3': row['col3'], 'col4': row['col4']}, ignore_index=True)
    #df_train.to_csv('df_train.csv', sep=',', encoding='utf-8')
    #df_val.to_csv('df_val.csv', sep=',', encoding='utf-8')

    return df_train,df_val

   #
    # # unique_pairs_list = list(df['col1'])
    # # pairs_names_pos_neg_df = pd.DataFrame(columns=['name', 'pos', 'neg'])
    # # for index,row in df.iterrows():
    # #     if (row['col4'] == -1):#positive pair
    # #         pairs_names_pos_neg_df = pairs_names_pos_neg_df.append ({'name': row['col1'], 'pos': 1, 'neg': 0}, ignore_index=True)
    # #     else:#negative pair
    # #         pairs_names_pos_neg_df = pairs_names_pos_neg_df.append ({'name': row['col1'], 'pos': 0, 'neg': 1}, ignore_index=True)
    # #         pairs_names_pos_neg_df = pairs_names_pos_neg_df.append ({'name': row['col3'], 'pos': 0, 'neg': 1}, ignore_index=True)
    # # pairs_names_pos_neg_df = pairs_names_pos_neg_df.groupby ('name').agg (['sum'], as_index=False)
    # # pairs_names_pos_neg_df.to_csv('pairs_pos_neg.csv', sep=',', encoding='utf-8')
    #
    # percentage = 0.2
    # unique_pairs_set = set(df['col1'])
    # for index, row in df.iterrows () :
    #     if (row['col4'] != -1):
    #         unique_pairs_set.add(row['col3'])
    # unique_pairs_list = list(unique_pairs_set)
    # random.shuffle(unique_pairs_list)
    # unique_pairs_len = len(unique_pairs_list)
    # unique_pairs_len_val = math.floor(percentage*unique_pairs_len)
    # unique_pairs_list_val = unique_pairs_list[0: unique_pairs_len_val]
    # unique_pairs_list_train = unique_pairs_list[unique_pairs_len_val : unique_pairs_len] #check startttttttttttttt unique_pairs_len_val
    # columns=['col1', 'col2', 'col3', 'col4']
    # list_new_for_df = []
    # for index, row in df.iterrows():
    #     if row['col1'] in unique_pairs_list_val:
    #         # df_new.append([{'col1': row['col1'], 'col2': row['col2'], 'col3': row['col3'], 'col4': row['col4']}], ignore_index=True)
    #         list_row = []
    #         list_row.append(row['col1'])
    #         list_row.append(row['col2'])
    #         list_row.append(row['col3'])
    #         list_row.append(row['col4'])
    #         list_new_for_df.append(list_row)
    # df_new = pd.DataFrame (list_new_for_df, columns=columns)
    # return df_new