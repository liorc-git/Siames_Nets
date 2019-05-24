import tensorflow as tf
import dataset_helpers as ds
import random
import numpy as np
import math
import pandas as pd

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

def train_model_net (X_train, Y_train, X_val, Y_val, iterations_num, batch_num, list_same, list_diff, siamese_model):#, support_set_size, final_momentum, momentum_slope, evaluate_each, model_name )
    """The function vlavulate haow many batches includes in train just that half will be from same and half will be from different
     after that we produce list of indexes for train anf produce new arrays (images and labels) which will be transferred to train on batch function
     """
    num_train_pairs = len(X_train[0])
    num_val_pairs = len (X_val[0])
    random.shuffle(list_same)
    random.shuffle(list_diff)
    len_list_same = len(list_same)
    len_list_diff = len(list_diff)
    batch_loop_num_half = int(batch_num/2)

    batch_loop_same = math.floor(len_list_same/batch_loop_num_half)
    batch_loop_diff = math.floor(len_list_diff/batch_loop_num_half)
    batch_loop_num = min(batch_loop_same, batch_loop_diff)
    # df_train_results = pd.DataFrame()
    df_validation_results = pd.DataFrame (columns=['epoch', 'loss', 'accuracy'])
    val_acc_prev = 0
    decrease_acc_num = 0#for consecutive decrease in accuracy
    for iter in range(iterations_num):
        for index in range(batch_loop_num):
            list_batch_indexes = []
            list_batch_indexes.extend(list_same[index*batch_loop_num_half:(index+1)*batch_loop_num_half])
            list_batch_indexes.extend(list_diff[index*batch_loop_num_half:(index+1)*batch_loop_num_half])
            random.shuffle(list_batch_indexes)
            X = [np.zeros ((batch_num, 250, 250, 1)) for i in range(2)]
            Y = np.zeros ((batch_num, 1))
            for index_batch, value in enumerate(list_batch_indexes):
                X[0][index_batch, :, :] = X_train[0][value]
                X[1][index_batch, :, :] = X_train[1][value]
                Y[index_batch] = Y_train[value]
            image_batch = X
            label_batch = Y
            train_loss, train_acc = siamese_model.train_on_batch(image_batch, label_batch)# not in useeeeeeeeee
        rem_same_pairs = len_list_same - (batch_loop_num*batch_loop_num_half)
        rem_diff_pairs = len_list_diff - (batch_loop_num*batch_loop_num_half)
        #need to check if we want to make train on the remain
        list_rem_batch_indexes = []
        list_rem_batch_indexes.append(list_same[-1*rem_same_pairs:])
        list_rem_batch_indexes.append(list_diff[-1*rem_diff_pairs:])
        random.shuffle(list_rem_batch_indexes)
        X = [np.zeros((batch_num, 250, 250, 1)) for i in range(2)]
        Y = np.zeros((batch_num, 1))
        for index_batch, value in enumerate(list_rem_batch_indexes):
            X[0][index_batch, :, :] = X_train[0][value]
            X[1][index_batch, :, :] = X_train[1][value]
            Y[index_batch] = Y_train[value]
        image_batch = X
        label_batch = Y
        train_loss, train_acc = siamese_model.train_on_batch (image_batch, label_batch)# not in useeeeeeeeee


        #evaluate for validation and check if there are 20 consecutive decrease in validation accuracy
        val_loss, val_acc = siamese_model.evaluate([X_val[0], X_val[1]], Y_val)
        df_validation_results = df_validation_results.append({'epoch': iter, 'loss': float(val_loss), 'accuracy': float(val_acc)}, ignore_index=True)
        if iterations_num % 20 == 0:
            print('Validation loss, Validation Accuracy at epoch %s: %s, %s' % (iter, float (val_loss), float(val_acc)))
        if iter > 0:
            val_acc_prev > val_acc
            decrease_acc_num += 1
            if decrease_acc_num == 20:
                break
        else:
            decrease_acc_num = 0
        val_acc_prev = val_acc
    return df_validation_results



        #we need to define new numpy array with the new indexes from batch_indexes and to transfer it to train on batches function
    #we need to transfer all the remaining indexes whic weren't included in the batch loop


    print(1)
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