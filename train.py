import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.regularizers import l2, l1
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten

def makeXY(rating_file, img_file):
    ratings = np.load(rating_file)
    imgs = np.load(img_file)
    X = np.zeros((len(ratings[0]), imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    Y = np.zeros(len(ratings[0]))

    for i, img_idx in enumerate(ratings[0]):
        X[i] = imgs[int(img_idx)]
        Y[i] = ratings[1][i]
    return X, Y

def build_conv_net(reg_param):
    """Simple convolutional NN"""
    model = Sequential()
    model.add(Conv2D(10, (5, 5), padding='same',
                     input_shape=(128, 128, 3),
                     kernel_regularizer=l2(reg_param)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(10, (3, 3), padding='same',
                     kernel_regularizer=l2(reg_param)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Need to flatten tensor output from conv layer to vector for dense layer
    model.add(Flatten())
    model.add(Dense(1, kernel_regularizer=l2(reg_param)))

    return model

if __name__ == '__main__':
    X, Y = makeXY('ratings1.npy', 'food_images.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    for k in range(-5, 0):
        print(k, '-------------------------------------------------------------')
        model = build_conv_net(10 ** k)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        history = model.fit(X, Y, batch_size=1, epochs=15, verbose=1)
                            #validation_data=(X_test, Y_test))
