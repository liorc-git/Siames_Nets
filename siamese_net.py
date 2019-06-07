import tensorflow as tf
import random
import numpy as np
import math
import pandas as pd
import datetime

def create_siamese_model(X_train):
    X_train_pair1 = tf.image.grayscale_to_rgb(X_train[0])
    X_train_pair2 = tf.image.grayscale_to_rgb (X_train[1])
    X_train_2 = []
    X_train_2.append(X_train_pair1)
    X_train_2.append(X_train_pair2)
    vggface = tf.keras.applications.vgg16.VGG16 () #if your image is different than 244*244, you should train your own classifier [include=false]
    siamese_net = tf.keras.models.Sequential ()
    for layer in vggface.layers:
        siamese_net.add(layer)
    siamese_net.pop()
    for layer in siamese_net.layers:
        layer.trainable=False

    twin_shape = X_train_2[0][0].shape
    twin_1_input = tf.keras.layers.Input(twin_shape)

    flatten_twin_1 = siamese_net(twin_1_input)
    twin_2_input = tf.keras.layers.Input(twin_shape)
    flatten_twin_2 = siamese_net(twin_2_input)

    # Calc L1 distance between twins
    dist_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    l1_dist = dist_layer([flatten_twin_1, flatten_twin_2])

    prediction = tf.keras.layers.Dense(units=1, activation='sigmoid')(l1_dist)
    model = tf.keras.models.Model(inputs=[twin_1_input, twin_2_input], outputs=prediction)

    return model


def train_model_net(X_train, Y_train, X_val, Y_val, X_test, y_test, iterations_num, batch_num, list_same, list_diff,
                        siamese_model):  # , support_set_size, final_momentum, momentum_slope, evaluate_each, model_name )
        """

        :param X_train: list of pairs with normalized pixels for each image in train dataset
        :param Y_train: label (1=identical, 0=different) for each pair n train dataset
        :param X_val: list of pairs with normalized pixels for each image in validation dataset
        :param Y_val: label (1=identical, 0=different) for each pair n validation dataset
        :param X_test: list of pairs with normalized pixels for each image in test dataset
        :param y_test: label (1=identical, 0=different) for each pair n test dataset
        :param iterations_num: max epochs if the stopping criteria doesn't exist
        :param batch_num: This is the batch size, number of pairs in one batch
        :param list_same: list with indexes of identical pairs in the train dataset
        :param list_diff: list with indexes of different pairs in the train dataset
        :param siamese_model: the model we use for train and evaluate
        :return: df_train_results (results from train), df_validation_results (results from validation), test_loss, test_acc (the final result for test after stopping criteria)
        """
        #"""The function evaluates how many batches includes in train just that half will be identical and half will be different
         #after that we produce list of indexes for train anf produce new arrays (images and labels) which will be transferred to train on batch function
         #"""
        # num_train_pairs = len(X_train[0])
        # num_val_pairs = len(X_val[0])

        #Adapt the validation and test pairs list to be with 3 channels , because input of RGB is expected by VGG16 model which is been used for our siamese model
        X_val_3channels = [tf.image.grayscale_to_rgb(X_val[0]),tf.image.grayscale_to_rgb(X_val[1])]
        X_test_3channels = [tf.image.grayscale_to_rgb(X_test[0]),tf.image.grayscale_to_rgb(X_test[1])]

        #calculate how many pairs will be from identical and different pairs=half of batch size
        len_list_same = len(list_same)
        len_list_diff = len(list_diff)
        batch_loop_num_half = int(batch_num / 2)

        #calculate number of iterations according to number of batch sizes
        batch_loop_same = math.floor(len_list_same / batch_loop_num_half)
        batch_loop_diff = math.floor(len_list_diff / batch_loop_num_half)
        batch_loop_num = min(batch_loop_same, batch_loop_diff)

        #produce dataframe with results
        df_train_results = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
        df_validation_results = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

        val_acc_prev = 0
        decrease_acc_num = 0  # for consecutive decrease in accuracy

        start = datetime.datetime.utcnow()#start time for checking running time for all epochs until convergence
        stop = 0
        for epoch in range(iterations_num):
            #shuffle each list indexes of identical and different pairs in order that ech batch will be different between epochs running
            random.shuffle(list_same)
            random.shuffle(list_diff)
            for index in range(batch_loop_num): #each batch has equal number of identical and different batches
                #produce list indexes for each batch and shuffle it - in order that the identical and different pairs will be learnt shuffled and not with convention of 1's and then 0's
                list_batch_indexes = []
                list_batch_indexes.extend(list_same[index * batch_loop_num_half:(index + 1) * batch_loop_num_half])
                list_batch_indexes.extend(list_diff[index * batch_loop_num_half:(index + 1) * batch_loop_num_half])
                random.shuffle(list_batch_indexes)
                #X and Y are arrays with pixels and 3 channels for batch train
                X = [np.zeros((batch_num, 250, 250, 3)) for i in range(2)]
                Y = np.zeros((batch_num, 1))
                for index_batch, value in enumerate(list_batch_indexes):#import the pixels accordint to index in train dataset (=value in loop)
                    X[0][index_batch, :, :] = tf.image.grayscale_to_rgb(X_train[0][value])
                    X[1][index_batch, :, :] = tf.image.grayscale_to_rgb(X_train[1][value])
                    Y[index_batch] = Y_train[value]
                image_batch = X
                label_batch = Y
                train_loss, train_acc = siamese_model.train_on_batch(image_batch, label_batch)
                df_train_results = df_train_results.append (
                    {'epoch' : epoch, 'loss' : float (train_loss), 'accuracy' : float (train_acc)}, ignore_index=True)
                print('Train loss, Train Accuracy at epoch %s, batch %s: %s, %s' % (epoch, index, float(train_loss), float(train_acc)))

                ## remains(for all pairs which aren't in batch loop)
                # rem_same_pairs = len_list_same - (batch_loop_num*batch_loop_num_half)
                # rem_diff_pairs = len_list_diff - (batch_loop_num*batch_loop_num_half)
                # # need to check if we want to make train on the remain
                # list_rem_batch_indexes = []
                # list_rem_batch_indexes.extend(list_same[-1 * rem_same_pairs:])
                # list_rem_batch_indexes.extend(list_diff[-1 * rem_diff_pairs:])
                # random.shuffle(list_rem_batch_indexes)
                # len_reamins = len(list_rem_batch_indexes)
                # X = [np.zeros((len_reamins, 250, 250, 3)) for i in range(2)]
                # Y = np.zeros((len_reamins, 1))
                # for index_batch, value in enumerate(list_rem_batch_indexes):
                #     X[0][index_batch, :, :] = tf.image.grayscale_to_rgb(X_train[0][value])
                #     X[1][index_batch, :, :] = tf.image.grayscale_to_rgb(X_train[1][value])
                #     Y[index_batch] = Y_train[value]
                # image_batch = X
                # label_batch = Y
                # # train_loss, train_acc = siamese_model.train_on_batch(image_batch, label_batch)  # not in useeeeeeeeee
                # # print('Train loss, Train Accuracy at epoch %s, final_batch: %s, %s' % (epoch, float(train_loss), float(train_acc)))



            # evaluate for validation and check if there are 20 consecutive decrease in validation accuracy
            val_loss, val_acc = siamese_model.evaluate([X_val_3channels[0], X_val_3channels[1]], Y_val)
            df_validation_results = df_validation_results.append(
                {'epoch': epoch, 'loss': float(val_loss), 'accuracy': float(val_acc)}, ignore_index=True)
            test_loss,test_acc = siamese_model.evaluate ([X_test_3channels[0], X_test_3channels[1]], y_test)
            #if epoch % 20 == 0:
            #    print('Validation loss, Validation Accuracy at epoch %s: %s, %s' % (epoch, float (val_loss), float (val_acc)))
            if epoch > 0:
                acc_decrease_per = (val_acc_prev - val_acc)/val_acc_prev
                if acc_decrease_per > 0.01:
                    decrease_acc_num += 1
                else:
                    decrease_acc_num = 0
                if decrease_acc_num >= 20:
                    stop = datetime.datetime.utcnow()
                    test_loss, test_acc = siamese_model.evaluate ([X_test_3channels[0], X_test_3channels[1]], y_test)
                    break
            val_acc_prev = val_acc
            conv_runtime = stop - start
        return df_train_results, df_validation_results, test_loss, test_acc, conv_runtime

