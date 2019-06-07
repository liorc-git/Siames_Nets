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

def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = -1 * y_true + 1
    return tf.keras.backend.mean((y_true) * tf.keras.backend.square(y_pred) + (1-y_true) *  tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0.0)))

optimizer = tf.optimizers.Adam (learning_rate=0.0005, decay=0.0,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-08,
                                     name='adam'
                                     )
siamese_model.compile (optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy'])#new_model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

iterations_num = 200
batch_num = 32
siamese_net.train_model_net(X_train, y_train, X_val, y_val, X_test, y_test, iterations_num, batch_num, list_same_train, list_diff_train, siamese_model)# insert X and YYYYYYYYYYYY

#predict
siamese_model.evaluate([X_test[0], X_test[1]], y_test)