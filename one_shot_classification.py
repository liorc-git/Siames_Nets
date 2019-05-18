import os
import tensorflow as tf
import dataset_helpers as ds
import siamese_net

jpg_path = './Datasets/lfwa_img_dirs'
directory = os.fsencode(jpg_path)
pairs_files = ['pairsDevTrain.txt', 'pairsDevTest.txt']
txt_files_path = './Datasets/pairs'

df_train_train, df_train_val = ds.split_train(txt_files_path, pairs_files[0])
df_test = ds.list_names_df(txt_files_path, pairs_files[1])

X_train, Y_train = ds.preprocess_data(directory, df_train_train, "Train")
X_test, Y_test = ds.preprocess_data(directory, df_test, "Test")


siamese_model = siamese_net.create_siamese_model(X_train)

siamese_model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

siamese_model.fit([X_train[0], X_train[1]], Y_train, epochs=5)

siamese_model.evaluate([X_test[0], X_test[1]], Y_test)


# # Define the optimizer and compile the model
# optimizer = Modified_SGD(
#     lr=self.learning_rate,
#     lr_multipliers=learning_rate_multipliers,
#     momentum=0.5)
#
# self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
#                    optimizer=optimizer)

