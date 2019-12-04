#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model        
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten

def plot_history_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.figure()
    plt.plot(hist.history['loss'],label="Training ")
    plt.plot(hist.history['val_loss'],label="Validation set")
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_history_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.figure()
    plt.plot(hist.history['categorical_accuracy'],label="Training set")
    plt.plot(hist.history['val_categorical_accuracy'],label="Validation set")
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    
#%%

num_pairs = 3000

pos_dir = '../data/16_8/positive_x/*'
neg_dir = '../data/16_8/negative_x/*'

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X = X[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X = np.concatenate((X,X_neg), axis=0)

for i in range(1,num_pairs):
    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_neg), axis=0)

print('X.shape:', X.shape)

# Create Y: Dock-(1,0), No dock-(0,1)
Y = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y.shape:', Y.shape)

#maybe 必要
# (サンプル数, channel, height, width)を(サンプル数, height, width, channel)に変換
# X_train = X_train.transpose([0, 2, 3, 1])
# X_test = X_test.transpose([0, 2, 3, 1])
# ⇨imshow関数で画像を出力するにあたっては､要素を(height, width, channel)の並び順に変換する必要がある

# 正規化処理
# X_train /= 255.0
# X_test /= 255.0

#%%

# ミニバッチに含まれるサンプル数を指定
batch_size = 50
# epoch数を指定
n_epoch = 10

#%%

# CNN
cnn = create_cnn(dim=(24,4,1))
print(cnn.summary())

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
cnn_hist = cnn.fit(X,Y,
               epochs=n_epoch,
               validation_split=0.1,
               verbose=1,
               batch_size=batch_size)

# ## 学習結果の確認
print('\nCNN Validation Loss:', cnn_hist.history['val_loss'][-1])
print('\nCNN Validation Accuracy:', cnn_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(cnn_hist)
plot_history_acc(cnn_hist)

#%%

mlp = create_mlp(dim=(24,4,1))
print(mlp.summary())

mlp.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['categorical_accuracy'])

# 学習を開始します。
print('Training model...')
mlp_hist = mlp.fit(X,Y,
                   epochs=n_epoch,
                   validation_split=0.1,
                   verbose=1,
                   batch_size=batch_size)

# ## 学習結果の確認
print('\nMLP Validation Loss:', mlp_hist.history['val_loss'][-1])
print('\nMLP Validation Accuracy:', mlp_hist.history['val_categorical_accuracy'][-1])
plot_history_loss(mlp_hist)
plot_history_acc(mlp_hist)

# In[22]:


# 性能指標を確認
from sklearn import metrics
print('accuracy: %.3f' % metrics.accuracy_score(y_test, model.predict(X_test).argmax(axis=1)))
print('recall: %.3f' % metrics.recall_score(y_test, model.predict(X_test).argmax(axis=1), average='macro'))
print('precision: %.3f' % metrics.precision_score(y_test, model.predict(X_test).argmax(axis=1), average='macro'))
print('f1_score: %.3f' % metrics.f1_score(y_test, model.predict(X_test).argmax(axis=1), average='macro'))

# In[25]:


#混合行列の出力
from sklearn.metrics import confusion_matrix as cm

# 混同行列きれいに出力する関数
def plot_cm(y_true, y_pred):
    confmat = cm(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xticks(np.arange(0, 5, 1))                               # x軸の目盛りを指定
    plt.yticks(np.arange(0, 5, 1))  
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
    
# 混同行列を出力
result = model.predict(X_test).argmax(axis=1)
plot_cm(y_test, result)


# In[26]:


print(label)


# In[27]:


# 予測が外れたtestデータを格納するリストを用意
errors = []

# すべてのtestデータをチェック
for i in range(len(y_test)):
    pred_1 = model.predict(X_test[i].reshape(1, 32, 32, 3)).argmax()
    if pred_1 != y_test[i]:
        # (index, 正解ラベル, 予測ラベル)をタプル形式で格納
        errors.append((i, label[y_test[i]], label[pred_1]))


# In[28]:


# 誤分類の総数を出力
len(errors)


# In[29]:


# 予測が外れた画像を集めて表示する
# 数が多いので3つだけ表示
for error_index, corr_label, pred_label in errors[:3]:
    show_test_sample_info(error_index)
    print(corr_label)
    print(pred_label)


# In[30]:


import keras.callbacks
import keras.backend.tensorflow_backend as KTF

# tensorFlowの形式に則り、sessionを保存
old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1) #訓練とテストでネットワーク構造が変わる時に必要。特段理由がなければひとまず入れる。

    # モデル構築は全く同じ
    model = Sequential()

    model.add(Conv2D(64, input_shape=(32, 32, 3),
                     kernel_size=(4, 4),
                     strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(128,
                     kernel_size=(4, 4),
                     strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(128,
                     kernel_size=(4, 4),
                     strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    # ミニバッチに含まれるサンプル数を指定
    batch_size = 500

    # epoch数を指定
    n_epoch = 2
    
    # ログファイルの保存先
    log_filepath = './cnn_log'
    
    # histogram_freqは出力頻度、1なら1epochごとに出力
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    cbks = [tb_cb]
    
    # callbacksパラメータが新しく追加される。Tensorboard用のCallbackは学習の直前と毎epochの終了時にCallされます。
    hist = model.fit(X_train,
                 Y_train,
                 epochs=n_epoch,
                 validation_data=(X_test, Y_test),
                 verbose=1,
                 batch_size=batch_size,
                 callbacks=cbks)
    
    
KTF.set_session(old_session)

