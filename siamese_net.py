import tensorflow as tf


def create_siamese_model(X_train):

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

    return model
