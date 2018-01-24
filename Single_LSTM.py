import cPickle as pickle

import deepdish as dd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, SGD


# ----------------------------------------------------------------------------------------------------------------------
# Helper Function that reads and splits data into train and test
# ----------------------------------------------------------------------------------------------------------------------
def data_preparation_UNBC():
    # Variables Initialization
    names = ['042-ll042', '043-jh043', '047-jl047', '048-aa048', '049-bm049', '052-dr052', '059-fn059', '064-ak064',
             '066-mg066', '080-bn080', '092-ch092', '095-tv095', '096-bg096', '097-gf097', '101-mg101', '103-jk103',
             '106-nm106', '107-hs107', '108-th108', '109-ib109', '115-jy115', '120-kz120', '121-vw121', '123-jh123',
             '124-dn124']

    test_names = names[18]
    print test_names


    vgg_features = dd.io.load('features/lstm_data_final.h5')
    with open('features/lstm_data.pkl', 'rb') as f:
        file = pickle.load(f)

    print len(vgg_features.keys())

    # Splitting Data
    for target_name in names[:1]:

        # Initialize Variables
        train_features = []
        train_labels = []
        train_weights = []

        test_features = []
        test_labels = []
        test_weights = []

        test_fnames = []
        train_fnames = []

        # Loop in 4 pain cluster (nopain, lowpain, mediumpain, and highpain)
        for k in ['C1', 'C2', 'C3', 'C4']:

            # Read the selected filenames in each cluster with labels
            filenames = file[k][0][:10]
            for i in range(len(filenames)):

                f = filenames[i]
                pic_num = int(f[-7:-4])
                tmp_vgg_features = []
                tmp_label = [0, 0, 0, 0]
                tmp_label[int(k[1:]) - 1] = 1
                tmp_weight = 1.0

                # Weight for each sample
                if int(k[1:]) == 4:
                    tmp_weight = 5.0
                elif int(k[1:]) == 3:
                    tmp_weight = 1.5

                # Extract the vgg features of a 15 frame sequence lasting with filenames[i]
                for num in range(pic_num - 15 + 1, pic_num + 1):
                    f = filenames[i][:-7] + str(num).zfill(3)
                    tmp_vgg_features.append(vgg_features[f[f.rfind("/")+1:]][0])

                # SPlitting train and test for each subject
                # if target_name[-5:] in filenames[i]:
                # if 1 == 2:
                print test_names
                if filenames[i][1:10] in test_names:
                    test_features.append(tmp_vgg_features)
                    test_labels.append(tmp_label)
                    test_weights.append(tmp_weight)
                    test_fnames.append(filenames[i])

                else:
                    train_features.append(tmp_vgg_features)
                    train_labels.append(tmp_label)
                    train_weights.append(tmp_weight)
                    train_fnames.append(filenames[i])

    # Save the data in data.npy file
    np.save('features/data.npy', np.array([train_features, train_labels, train_weights,
                                  test_features, test_labels, test_weights, train_fnames, test_fnames]))


# ----------------------------------------------------------------------------------------------------------------------
# Bulid LSTM Model and train it in KERAS
# ----------------------------------------------------------------------------------------------------------------------
def lstm(model_name, nb_hidden, nb_classes, input_shape, batch_size, nb_epoch, data_file_path):

    # Loading Train and Test Data
    print 'Loading Data ... '
    data = np.load(data_file_path)
    X_train = np.array(data[0])
    Y_train = np.array(data[1])
    W_train = np.array(data[2])
    X_test = np.array(data[3])
    Y_test = np.array(data[4])
    W_test = np.array(data[5])
    train_fnames = data[6]
    test_fnames = data[7]

    print 'Data is Loaded!'

    # Defining Callbacks for Models
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                 ModelCheckpoint('LSTM_model_' + model_name + '.h5', monitor='val_acc', save_best_only=True, verbose=0)]

    adam = Adam(lr=1e-5, decay=1e-6) #best lr = 3*0.0001
    sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)

    # Build a simple LSTM Model
    if model_name == '1':
        print 'Building Single LSTM Model! with only one lstm cell'
        model = Sequential()
        model.add(LSTM(nb_hidden, return_sequences=False, input_shape=input_shape, dropout=0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        # Compiling and Fitting the model
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, sample_weight=W_train, batch_size=batch_size, verbose=1, epochs=nb_epoch,
                  validation_data=(X_test, Y_test, W_test), shuffle=True)

    elif model_name == '2':
        print 'Building Single LSTM Model! with only one lstm cell'
        model = Sequential()
        model.add(LSTM(256, dropout=0.2, input_shape=(32, 15, 512)))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        # Compiling and Fitting the model
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, sample_weight=W_train, batch_size=batch_size, verbose=1, epochs=nb_epoch,
                  validation_data=(X_test, Y_test, W_test), shuffle=True, callbacks=callbacks)

    elif model_name == '3':
        model = Sequential()
        model.add(LSTM(nb_hidden, dropout=0.5, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(nb_hidden, return_sequences=True))
        model.add(LSTM(nb_hidden, dropout=0.5, return_sequences=True))
        model.add(LSTM(nb_hidden, return_sequences=True))
        model.add(LSTM(nb_hidden, dropout=0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, sample_weight=W_train, batch_size=batch_size, verbose=1, epochs=nb_epoch,
                         validation_data=(X_test, Y_test, W_test), shuffle=True, callbacks=callbacks)

    # Predicting Labels fot Test Samples
    Pred_test = model.predict(X_test)
    Pred_classes = np.argmax(Pred_test, axis=1)
    True_classes = np.argmax(Y_test, axis=1)
    print Pred_test
    print Pred_classes
    print True_classes
    print 'accuracy!'
    print sum(Pred_classes==True_classes)

    with open('history_model_' + model_name + '.pkl', 'wb') as f:
        pickle.dump(hist.history, f)

if __name__ == '__main__':
    # Variables
    nb_classes = 4
    nb_hidden = 128
    input_shape = (15, 512)
    batch_size = 32
    nb_epoch = 300
    model_name = '1'


    # Getting the data
    # data_preparation_UNBC()

    data_file_path = 'features/data.npy'
    lstm(model_name, nb_hidden, nb_classes, input_shape, batch_size, nb_epoch, data_file_path)






