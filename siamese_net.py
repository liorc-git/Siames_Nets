import tensorflow as tf
import dataset_helpers as ds


def create_siamese_model(X_train):

    convolutional_net = tf.keras.models.Sequential()

    convolutional_net.add(tf.keras.layers.Conv2D(filters=64,
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
    convolutional_net.add(tf.keras.layers.MaxPool2D())

    convolutional_net.add(tf.keras.layers.Conv2D(filters=128,
                                                 kernel_size=(7, 7),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_2'))
    convolutional_net.add(tf.keras.layers.MaxPool2D())

    convolutional_net.add(tf.keras.layers.Conv2D(filters=128,
                                                 kernel_size=(4, 4),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_3'))
    convolutional_net.add(tf.keras.layers.MaxPool2D())

    convolutional_net.add(tf.keras.layers.Conv2D(filters=256,
                                                 kernel_size=(4, 4),
                                                 kernel_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                                                 use_bias=True,
                                                 bias_initializer=
                                                    tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1e-2),
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                                 name='Conv_4'))

    convolutional_net.add(tf.keras.layers.Flatten())
    convolutional_net.add(
        tf.keras.layers.Dense(units=4096,
                              activation='sigmoid',
                              kernel_initializer=
                                tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-2),
                              use_bias=True,
                              bias_initializer=
                                tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=2*(1e-1)),
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                              name='Dense_1'))

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

def train_model_net (train_data, val_data, iterations_num, support_set_size,
                            final_momentum, momentum_slope, evaluate_each,
                            model_name ) :
    """for each epoch: (stopping criteria 200 epochs or calidation error not decrease for 20 epochs)
    1.for each iteration:
    a. get batch
    b. train model --> train_loss, train_accuracy = self.model.train_on_batch (
            images, labels)

    c. update learning rate=n(T)=0.99*n(T-1)
    momentum is intialized to 0.5 anf increase until 1
    d. each X epochs we calculate validation error

    return parameters

    """
def test_model_net (test, parameters):
    """get the test and put it on model"""


    """ Train the Siamese net
    This is the main function for training the siamese net.
    In each every evaluate_each train iterations we evaluate one-shot tasks in
    validation and evaluation set. We also write to the log file.
    Arguments:
        number_of_iterations: maximum number of iterations to train.
        support_set_size: number of characters to use in the support set
            in one-shot tasks.
        final_momentum: mu_j in the paper. Each layer starts at 0.5 momentum
            but evolves linearly to mu_j
        momentum_slope: slope of the momentum evolution. In the paper we are
            only told that this momentum evolves linearly. Because of that I
            defined a slope to be passed to the training.
        evaluate each: number of iterations defined to evaluate the one-shot
            tasks.
        model_name: save_name of the model
    Returns:
        Evaluation Accuracy
    """

    # First of all let's divide randomly the 30 train alphabets in train
    # and validation with 24 for training and 6 for validation

    # Variables that will store 100 iterations losses and accuracies
    # after evaluate_each iterations these will be passed to tensorboard logs








    train_losses = np.zeros (shape=(evaluate_each))
    train_accuracies = np.zeros (shape=(evaluate_each))
    count = 0
    earrly_stop = 0
    # Stop criteria variables
    best_validation_accuracy = 0.0
    best_accuracy_iteration = 0
    validation_accuracy = 0.0

    #In Each iteration we split train and take the batch and make on it train,
    for iteration in range(iterations_num) :

        # train set
        images, labels = self.omniglot_loader.get_train_batch ()
        train_loss, train_accuracy = self.model.train_on_batch (
            images, labels)

        # Decay learning rate 1 % per 500 iterations (in the paper the decay is
        # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
        if (iteration + 1)%500 == 0 :
            K.set_value (self.model.optimizer.lr, K.get_value (
                self.model.optimizer.lr)*0.99)
        if K.get_value (self.model.optimizer.momentum) < final_momentum :
            K.set_value (self.model.optimizer.momentum, K.get_value (
                self.model.optimizer.momentum) + momentum_slope)

        train_losses[count] = train_loss
        train_accuracies[count] = train_accuracy

        # validation set
        count += 1
        print ('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f'%
               (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value (
                   self.model.optimizer.lr)))

        # Each 100 iterations perform a one_shot_task and write to tensorboard the
        # stored losses and accuracies
        if (iteration + 1)%evaluate_each == 0 :
            number_of_runs_per_alphabet = 40
            # use a support set size equal to the number of character in the alphabet
            validation_accuracy = self.omniglot_loader.one_shot_test (
                self.model, support_set_size, number_of_runs_per_alphabet, is_validation=True)

            self._write_logs_to_tensorboard (
                iteration, train_losses, train_accuracies,
                validation_accuracy, evaluate_each)
            count = 0

            # Some hyperparameters lead to 100%, although the output is almost the same in
            # all images.
            if (validation_accuracy == 1.0 and train_accuracy == 0.5) :
                print ('Early Stopping: Gradient Explosion')
                print ('Validation Accuracy = ' +
                       str (best_validation_accuracy))
                return 0
            elif train_accuracy == 0.0 :
                return 0
            else :
                # Save the model
                if validation_accuracy > best_validation_accuracy :
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration

                    model_json = self.model.to_json ()

                    if not os.path.exists ('./models') :
                        os.makedirs ('./models')
                    with open ('models/' + model_name + '.json', "w") as json_file :
                        json_file.write (model_json)
                    self.model.save_weights ('models/' + model_name + '.h5')

        # If accuracy does not improve for 10000 batches stop the training
        if iteration - best_accuracy_iteration > 10000 :
            print (
                'Early Stopping: validation accuracy did not increase for 10000 iterations')
            print ('Best Validation Accuracy = ' +
                   str (best_validation_accuracy))
            print ('Validation Accuracy = ' + str (best_validation_accuracy))
            break

    print ('Trained Ended!')
    return best_validation_accuracy