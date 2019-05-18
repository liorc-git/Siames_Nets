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
        output: jpg normalized pixels as np array
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
jpg_path = './Datasets/lfwa_img_dirs'
directory = os.fsencode(jpg_path)

for folder in os.listdir(directory):
    folder_person_path = join(directory, folder)
    folder_person_path_name = os.fsdecode(folder_person_path)
    for jpg_file in os.listdir(folder_person_path_name):
        file = join(folder_person_path_name, jpg_file)
        file_pixels = get_image_pixels(file)
        # print(file_pixels)


""" X and Y are np arrays. 
    X[0] holds person a pixles X[1] holds person b pixles Y holds the lable. 
    All match the index (represents pair id)
"""
def preprocess_data(directory, txt_files_path, file_name):
    """ input: directory-jpg file directory
               txt_files_path-train and test txt files path
    """
    #convert float numbers of the 4th column (exist to no match pairs) to int and all nan to '-1'
    path_txt = join(txt_files_path, file_name)
    df = pd.read_csv(path_txt, names=['col1', 'col2', 'col3', 'col4'], skiprows=1, sep='\t')
    df['col4'].fillna(-1, inplace=True)
    df['col4'] = df['col4'].astype('int64')
    list_names = []
    pairs = df.shape[0]
    X = [np.zeros((pairs, 250, 250, 1)) for i in range (2)]
    Y = np.zeros((pairs, 1))

    for index, row in df.iterrows():
        list_names.append (row['col1'])
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

    set_names = set(list_names)
    return ("finish_data_processing", X,Y)

txt_files_path = './Datasets/pairs'
txt_file_name_train = 'pairsDevTrain.txt'
txt_file_name_test = 'pairsDevTest.txt'

res_train = preprocess_data(directory, txt_files_path, txt_file_name_train)
print(res_train[0] + " train")
X_train = res_train[1]
Y_train = res_train[2]
res_test = preprocess_data(directory, txt_files_path, txt_file_name_test)
print(res_test[0] + " test")
X_test = res_test[1]
Y_test = res_test[2]


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

