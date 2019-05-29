import dataset_helpers as ds
import siamese_net
import tensorflow as tf


jpg_path = './Datasets/lfwa_img_dirs/'
pairs_files = ['pairsDevTrain.txt', 'pairsDevTest.txt']
txt_files_path = './Datasets/pairs/'

df_train  = ds.read_data(txt_files_path + pairs_files[0])
df_test = ds.read_data(txt_files_path + pairs_files[1])
df_train_train, df_train_val = ds.split_train(df_train)

X_train, y_train, list_same_train, list_diff_train = ds.preprocess_data(jpg_path, df_train_train, "Train")
X_val, y_val = ds.preprocess_data(jpg_path, df_train_val, "Val")
X_test, y_test = ds.preprocess_data(jpg_path, df_test, "Test")

siamese_model = siamese_net.create_siamese_model(X_train)

# optimizer = tf.optimizers.SGD(  #tf.keras.optimizers.Adam
#         learning_rate=1e-4, # TODO layer wise learning_rates,
#         decay=0.99,
#         momentum=0.5, # TODO changing momentum
#         name='Momentum'
#     )


optimizer = tf.optimizers.Adam( learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-07,
    name='Adam'
)

siamese_model.compile(optimizer=optimizer,
               loss='binary_crossentropy',
               metrics=['accuracy'])

iterations_num = 200
batch_num = 128
siamese_net.train_model_net(X_train, y_train, X_val, y_val, iterations_num, batch_num, list_same_train, list_diff_train, siamese_model)# insert X and YYYYYYYYYYYY

#predict
siamese_model.evaluate([X_test[0], X_test[1]], y_test)
