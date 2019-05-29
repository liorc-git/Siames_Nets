import tensorflow as tf
import dataset_helpers as ds
import random
import numpy as np
import math
import pandas as pd

def create_siamese_model(X_train):

    siamese_net = tf.keras.models.Sequential()

    siamese_net.add(tf.keras.layers.Conv2D(filters=32, #64
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

    siamese_net.add(tf.keras.layers.Conv2D(filters=64, #128
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

    siamese_net.add(tf.keras.layers.Conv2D(filters=64, #128
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

    siamese_net.add(tf.keras.layers.Conv2D(filters=128, #256
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
    siamese_net.add(tf.keras.layers.Dense(units=512, #4096
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


def train_model_net(X_train, y_train, X_val, y_val, iterations_num, batch_size, list_same, list_diff,
                        siamese_model):

        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        batch_size_per_class = math.floor(batch_size /2)
        df_train_res = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
        df_val_res = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

        for epoch in range(iterations_num):
            batch_num = 1
            while pos.any() or neg.any():
                pos_batch = random.choices(pos, k=min(batch_size_per_class, len(pos)))
                pos = np.array([x for x in pos if x not in pos_batch])
                neg_batch = random.choices(neg, k=min(batch_size_per_class, len(neg)))
                neg = np.array([x for x in neg if x not in neg_batch])
                batch_idx = pos_batch + neg_batch
                X = [X_train[0][batch_idx], X_train[1][batch_idx]]
                y = y_train[batch_idx]

                train_loss, train_acc = siamese_model.train_on_batch(X, y)
                df_train_res = df_train_res.append(
                    {'epoch': epoch, 'loss': float(train_loss), 'accuracy': float(train_acc)}, ignore_index=True)
                print('Train loss, Train Accuracy at epoch %s, batch %s: %s, %s' % (epoch, batch_num,
                                                                     float(train_loss), float(train_acc)))
                batch_num += 1

            val_loss, val_acc = siamese_model.evaluate([X_val[0], X_val[1]], y_val)
            df_val_res = df_val_res.append(
                {'epoch': epoch, 'loss': float(val_loss), 'accuracy': float(val_acc)}, ignore_index=True)
            print('Validation loss, Validation Accuracy at epoch %s: %s, %s' % (epoch, float (val_loss), float (val_acc)))

            if epoch % 20 == 0 and epoch != 0:

                if (df_val_res[df_val_res[epoch] == epoch]['accuracy']/
                    df_val_res[df_val_res[epoch] == epoch-20]['accuracy'])<1.01:
                        break

        return df_val_res


