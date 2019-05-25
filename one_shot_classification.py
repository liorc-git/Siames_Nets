import os
import dataset_helpers as ds
import siamese_net

jpg_path = './Datasets/lfwa_img_dirs'
directory = os.fsencode(jpg_path)
pairs_files = ['pairsDevTrain.txt', 'pairsDevTest.txt']
txt_files_path = './Datasets/pairs'

df_train_train, df_train_val = ds.split_train(txt_files_path, pairs_files[0])
df_test = ds.list_names_df(txt_files_path, pairs_files[1])

X_train, Y_train, list_same_train, list_diff_train = ds.preprocess_data(directory, df_train_train, "Train")
X_val, Y_val, list_same_val, list_diff_val = ds.preprocess_data(directory, df_train_val, "Train")# check what should be last stringggggg
X_test, Y_test, list_same_test, list_diff_test = ds.preprocess_data(directory, df_test, "Test")


siamese_model = siamese_net.create_siamese_model(X_train)

init_momentum = 0.5
learning_rates = {'conv_1': 1e-4, 'conv_2': 1e-4, 'conv_3': 1e-4, 'conv_4': 1e-4, 'dense_1': 1e-4}

optimizer = siamese_net.create_siamese_optimizer()

siamese_model.compile(optimizer=optimizer,
               loss='binary_crossentropy',
               metrics=['accuracy'])

siamese_model.fit([X_train[0], X_train[1]], Y_train, epochs=5)


# siamese_model.fit([X_train[0], X_train[1]], Y_train, epochs=5)
# siamese_model.evaluate([X_test[0], X_test[1]], Y_test)

iterations_num = 200
batch_num = 16
siamese_net.train_model_net(X_train, Y_train, X_val, Y_val, iterations_num, batch_num, list_same_train, list_diff_train, siamese_model)# insert X and YYYYYYYYYYYY

#predict
siamese_model.evaluate([X_test[0], X_test[1]], Y_test)
