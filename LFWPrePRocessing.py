from PIL import Image
import numpy as np
import os
from os.path import join
import pandas as pd
import random
import math
import tensorflow as tf

def get_image_pixels ( file_path ) :
    """
        input: jpg file path
        output: jpg normalized pixels
    """

    # load the image
    face_image = Image.open (file_path)
    # summarize some details about the image
    # print(face_image.format)
    # print(face_image.mode)
    # print(face_image.size)

    # produce pixels for face_image
    pixels = np.asarray (face_image)
    # normalize pixels to the range 0-1
    pixels = pixels.astype ('float32')
    pixels /= 255.0
    # confirm the normalization
    # print ('Min: %.3f, Max: %.3f'%(pixels.min (), pixels.max ()))
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

#jpg files path
jpg_path = 'C:\\Users\\rtrag\\Desktop\\DataScience\\BGU\\semester1_spring_2019\\DeepLearning\\Assignments\\Assignment2\\lfwa\\lfw2\\lfw2'
#jpg_path = './Datasets/lfwa_img_dirs'
directory = os.fsencode(jpg_path)

####################3check if we need to delte the following for loop
for folder in os.listdir(directory):
    folder_person_path = join(directory, folder)
    folder_person_path_name = os.fsdecode(folder_person_path)
    for jpg_file in os.listdir(folder_person_path_name):
        file = join(folder_person_path_name, jpg_file)
        file_pixels = get_image_pixels(file)
        # print(file_pixels)


""" X and Y are dictionaries
    X: Each pair has key: id and dictionary with the pair
    pair dictionary has key:name and value:pixles
    Y: dictionary with key: id and value: 1 (match) or 0(not match)
"""
def preprocess_data(directory, df):
    """ input: directory-jpg file directory
               txt_files_path-train and test txt files path
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

    return ("finish_data_processing", X,Y)

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
    df_train.to_csv('df_train.csv', sep=',', encoding='utf-8')
    df_val.to_csv('df_val.csv', sep=',', encoding='utf-8')
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

txt_files_path = 'C:\\Users\\rtrag\\Desktop\\DataScience\\BGU\\semester1_spring_2019\\DeepLearning\\Assignments\\Assignment2'
#txt_files_path = './Datasets/pairs'
txt_file_name_train = 'pairsDevTrain.txt'
txt_file_name_test = 'pairsDevTest.txt'

df_train = list_names_df(txt_files_path, txt_file_name_train)
df_test = list_names_df(txt_files_path, txt_file_name_test)
df_train_train,df_train_val = split_train(txt_files_path, txt_file_name_train)


res_train = preprocess_data(directory, df_train)
print(res_train[0] + " train")
X_train = res_train[1]
Y_train = res_train[2]
res_test = preprocess_data(directory, df_test)
print(res_test[0] + " test")
X_test = res_test[1]
Y_test = res_test[2]

res_train_train = preprocess_data(directory, df_train_train)
print(res_train_train[0] + " train_train")
X_train_train = res_train_train[1]
Y_train_train = res_train_train[2]

res_train_val = preprocess_data(directory, df_train_val)
print(res_train_val[0] + " train_val")
X_train_val = res_train_val[1]
Y_train_val = res_train_val[2]



#
# # def train_split(X_train,Y_train)
#
# """ 2 methods for train validation test:
#     choose 20% randomly from the train pairs fo validation pairs and 80% will be train pairs
#     choose 20% randomly distinct names from train set for validation
# """
# X_train_train = {}
# X_train_val = {}
# Y_train_train = {}
# Y_train_val = {}
# def split(method):
#     if (method == 'records'):
#         # method1 - split train, validation
#         keys = list(X_train.keys())
#         shuffled_keys = random.shuffle(keys)
#         key_len = len(keys)
#         for index, key in enumerate(keys):
#             if (index < 0.2*key_len):
#                 X_train_val[key] = X_train[key]
#                 Y_train_val[key] = Y_train[key]
#             else:
#                 X_train_train[key] = X_train[key]
#                 Y_train_train[key] = Y_train[key]
#     else:
#         #method2 - split train, validation
#         train_names_list = list(res_train[1])
#         train_names_list_len = len(train_names_list)
#         random.shuffle(train_names_list)
#         print(train_names_list_len)
#
#         for key, val in X_train.items():
#             per_amount = 0.2*train_names_list_len
#             per_amount = math.floor (per_amount)
#             val_names_list = train_names_list[0:per_amount]
#             for key_sub, val_sub in val.items():
#                 i = 0
#                 if (i == 0):
#                     if (key_sub in val_names_list):
#                         X_train_val[key] = val
#                         Y_train_val[key] = Y_train[key]
#                     else:
#                         X_train_train[key] = val
#                         Y_train_train[key] = Y_train[key]
#                 i+=1
#     return ('Train data is splitted')
# # method1='records', method2='names'
# res_split = split('names')
# print(res_split)

#######################################################


convolutional_net = tf.keras.models.Sequential()

convolutional_net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
                             activation='relu',
                             input_shape=X_train[0][0].shape,
                             kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                             name='Conv1'))
convolutional_net.add(tf.keras.layers.MaxPool2D())

convolutional_net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7),
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                             name='Conv2'))
convolutional_net.add(tf.keras.layers.MaxPool2D())

convolutional_net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                             name='Conv3'))
convolutional_net.add(tf.keras.layers.MaxPool2D())

convolutional_net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4),
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                             name='Conv4'))

convolutional_net.add(tf.keras.layers.Flatten())
convolutional_net.add(
    tf.keras.layers.Dense(units=4096, activation='sigmoid',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
          name='Dense1'))

# Now the pairs of images
input_image_1 = tf.keras.layers.Input(X_train[0][0].shape)
input_image_2 = tf.keras.layers.Input(X_train[0][0].shape)

encoded_image_1 = convolutional_net(input_image_1)
encoded_image_2 = convolutional_net(input_image_2)

# L1 distance layer between the two encoded outputs
# One could use Subtract from Keras, but we want the absolute value
l1_distance_layer = tf.keras.layers.Lambda(
    lambda tensors: tf.abs(tensors[0] - tensors[1]))
l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

# Same class or not prediction
prediction = tf.keras.layers.Dense(units=1, activation='sigmoid')(l1_distance)
model = tf.keras.models.Model(
    inputs=[input_image_1, input_image_2], outputs=prediction)


model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([X_train[0], X_train[1]], Y_train, epochs=5)

model.evaluate([X_test[0], X_test[1]], Y_test)


# # Define the optimizer and compile the model
# optimizer = Modified_SGD(
#     lr=self.learning_rate,
#     lr_multipliers=learning_rate_multipliers,
#     momentum=0.5)
#
# self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
#                    optimizer=optimizer)

