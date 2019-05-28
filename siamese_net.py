import tensorflow as tf
import dataset_helpers as ds
import random
import numpy as np
import math
import pandas as pd

def create_siamese_model(X_train):

    siamese_net = tf.keras.models.Sequential()

    siamese_net.add(tf.keras.layers.Conv2D(filters=64,
                                                 kernel_size=(10, 10),
                                                 kernel_initializer =
                                                    tf.keras.initializers.TruncatedNormal(mean = 0 ,stddev=1e-2),
                                                 use_bias = True,
                                                 bias_initializer =
                                                    tf.keras.initializers.TruncatedNormal(mean = 0.5 ,stddev=1e-2),
                                                 activation='relu',
                                                 input_shape=X_train[0][0].shape,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_1'))
    siamese_net.add(tf.keras.layers.MaxPool2D())

    siamese_net.add(tf.keras.layers.Conv2D(filters=128,
                                                 kernel_size=(7, 7),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_2'))
    siamese_net.add(tf.keras.layers.MaxPool2D())

    siamese_net.add(tf.keras.layers.Conv2D(filters=128,
                                                 kernel_size=(4, 4),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_3'))
    siamese_net.add(tf.keras.layers.MaxPool2D())

    siamese_net.add(tf.keras.layers.Conv2D(filters=256,
                                                 kernel_size=(4, 4),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_4'))

    siamese_net.add(tf.keras.layers.Flatten())
    siamese_net.add(tf.keras.layers.Dense(units=500, #4096 units
                              activation='sigmoid',
                              kernel_initializer=
                                tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                              use_bias=True,
                              bias_initializer=
                                tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=2*(1e-1)),
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              name='Dense_1'))

    twin_1_input = tf.keras.layers.Input(X_train[0][0].shape)
    twin_2_input = tf.keras.layers.Input(X_train[0][0].shape)

    flatten_twin_1 = siamese_net(twin_1_input)
    flatten_twin_2 = siamese_net(twin_2_input)

    # Calc L1 distance between twins
    dist_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    l1_dist = dist_layer([flatten_twin_1, flatten_twin_2])

    prediction = tf.keras.layers.Dense(units=1, activation='sigmoid')(l1_dist)
    model = tf.keras.models.Model(inputs=[twin_1_input, twin_2_input], outputs=prediction)


    return model


def create_siamese_optimizer(init_momentum, learning_rates):
    return tf.keras.optimizers.Adam(  #tf.optimizers.SGD
        learning_rate=1e-4, # TODO layer wise learning_rates,
        decay=0.99,
        momentum=init_momentum, # TODO changing momentum
        name='Momentum'
    )

def train_model_net(X_train, Y_train, X_val, Y_val, iterations_num, batch_num, list_same, list_diff,
                        siamese_model):  # , support_set_size, final_momentum, momentum_slope, evaluate_each, model_name )
        """The function vlavulate haow many batches includes in train just that half will be from same and half will be from different
         after that we produce list of indexes for train anf produce new arrays (images and labels) which will be transferred to train on batch function
         """
        num_train_pairs = len(X_train[0])
        num_val_pairs = len(X_val[0])

        len_list_same = len(list_same)
        len_list_diff = len(list_diff)
        batch_loop_num_half = int(batch_num / 2)

        batch_loop_same = math.floor(len_list_same / batch_loop_num_half)
        batch_loop_diff = math.floor(len_list_diff / batch_loop_num_half)
        batch_loop_num = min(batch_loop_same, batch_loop_diff)
        # df_train_results = pd.DataFrame()
        df_validation_results = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
        val_acc_prev = 0
        decrease_acc_num = 0  # for consecutive decrease in accuracy
        for epoch in range(iterations_num):
            random.shuffle (list_same)
            random.shuffle (list_diff)
            for index in range(batch_loop_num):
                list_batch_indexes = []
                list_batch_indexes.extend(list_same[index * batch_loop_num_half:(index + 1) * batch_loop_num_half])
                list_batch_indexes.extend(list_diff[index * batch_loop_num_half:(index + 1) * batch_loop_num_half])
                random.shuffle(list_batch_indexes)
                X = [np.zeros((batch_num, 250, 250, 1)) for i in range(2)]
                Y = np.zeros((batch_num, 1))
                for index_batch, value in enumerate(list_batch_indexes):
                    X[0][index_batch, :, :] = X_train[0][value]
                    X[1][index_batch, :, :] = X_train[1][value]
                    Y[index_batch] = Y_train[value]
                image_batch = X
                label_batch = Y
                siamese_model.fit(image_batch, label_batch, verbose=2)
                train_loss, train_acc = siamese_model.train_on_batch(image_batch, label_batch)# not in useeeeeeeeee
                print('Train loss, Train Accuracy at epoch %s, batch %s: %s, %s' % (epoch, index, float(train_loss), float(train_acc)))


            # remains(for all pairs which aren't in batch loop)
            rem_same_pairs = len_list_same - (batch_loop_num*batch_loop_num_half)
            rem_diff_pairs = len_list_diff - (batch_loop_num*batch_loop_num_half)
            # need to check if we want to make train on the remain
            list_rem_batch_indexes = []
            list_rem_batch_indexes.extend(list_same[-1 * rem_same_pairs:])
            list_rem_batch_indexes.extend(list_diff[-1 * rem_diff_pairs:])
            random.shuffle(list_rem_batch_indexes)
            len_reamins = len(list_rem_batch_indexes)
            X = [np.zeros((len_reamins, 250, 250, 1)) for i in range(2)]
            Y = np.zeros((len_reamins, 1))
            for index_batch, value in enumerate(list_rem_batch_indexes):
                X[0][index_batch, :, :] = X_train[0][value]
                X[1][index_batch, :, :] = X_train[1][value]
                Y[index_batch] = Y_train[value]
            image_batch = X
            label_batch = Y
            train_loss, train_acc = siamese_model.train_on_batch(image_batch, label_batch)  # not in useeeeeeeeee
            print('Train loss, Train Accuracy at epoch %s, final_batch: %s, %s' % (epoch, float(train_loss), float(train_acc)))
            # evaluate for validation and check if there are 20 consecutive decrease in validation accuracy
            val_loss, val_acc = siamese_model.evaluate([X_val[0], X_val[1]], Y_val)
            df_validation_results = df_validation_results.append(
                {'epoch': epoch, 'loss': float(val_loss), 'accuracy': float(val_acc)}, ignore_index=True)
            if epoch % 20 == 0:
                print('Validation loss, Validation Accuracy at epoch %s: %s, %s' % (epoch, float (val_loss), float (val_acc)))
            if epoch > 0:
                acc_decrease_per = (val_acc_prev - val_acc)/val_acc_prev
                if acc_decrease_per > 0.01:
                    decrease_acc_num += 1
                else:
                    decrease_acc_num = 0
                if decrease_acc_num >= 20:
                    break
            val_acc_prev = val_acc
        return df_validation_results

        """for each epoch: (stopping criteria 200 epochs or calidation error not decrease for 20 epochs)
        1.for each iteration:
        a. get batch
        b. train model --> train_loss, train_accuracy = self.model.train_on_batch (
                images, labels)
        c. update learning rate=n(T)=0.99*n(T-1)
        momentum is intialized to 0.5 anf increase until 1
        d. each X epochs we calculate validation error siamese_model.evaluate([X_test[0], X_test[1]], Y_test) --> loss , accuracy
        return df_results and the model id global
        """
