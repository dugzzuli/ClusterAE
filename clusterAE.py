from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import KMeans

# define the function


def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    np.savetxt("FCN_acc.txt", acc)
    np.savetxt("FCN_val_acc.txt", val_acc)
    np.savetxt("FCN_val_loss.txt", val_loss)
    np.savetxt("FCN_lossc.txt", loss)
    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def randIndex(truth, predicted):
    """
    The function is to measure similarity between two label assignments
    truth: ground truth labels for the dataset (1 x 1496)
    predicted: predicted labels (1 x 1496)
    """
    if len(truth) != len(predicted):
        print("different sizes of the label assignments")
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel-1):
        for j in range(i+1,sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same +=1
            count += 1
    return (agree_same+disagree_same)/float(count)


nb_epochs = 5000


# flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics',
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
# 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']
rootDir='./UCR_TS_Archive_2015/'
flist = ['Adiac']
for each in flist:
    fname = each
    x_train, y_train = readucr(rootDir+fname+'/'+fname+'_TRAIN')

    nb_classes = len(np.unique(y_train))
    batch_size = min(x_train.shape[0]/10, 16)

    latent_dim = nb_classes

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_train = x_train.reshape(x_train.shape + (1, 1,))
    arr = x_train.shape + (1, 1,)

    print(arr)

    inputs = keras.layers.Input(x_train.shape[1:])
    x = inputs

    x = Conv2D(filters=128, kernel_size=3, strides=1, border_mode='same')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=1, border_mode='same')(x)

    shape = K.int_shape(x)

    x = Flatten()(x)

    latent = Dense(latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2DTranspose(filters=256, kernel_size=3, strides=1,
                        activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=1,
                        activation='relu', padding='same')(x)

    x = Conv2DTranspose(filters=1,
                        kernel_size=3,
                        padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    autoencoder.compile(loss='mse', optimizer='adam')

    # Train the autoencoder
    autoencoder.fit(x_train,
                    x_train,
                    epochs=1,
                    batch_size=batch_size)
    x_encode = encoder.predict(x_train)
    print(np.shape(x_encode))

    np.savetxt("./data/alica.txt", x_encode, fmt="%.2e")

    kmeans = KMeans(n_clusters=nb_classes, random_state=0).fit(x_encode)
    pre_train = kmeans.labels_
    # y_train
    import sklearn
    score = sklearn.metrics.normalized_mutual_info_score(y_train, pre_train)
    print("真实标签:")
    print(y_train)
    print("预测标签:")
    print(pre_train)


    print("NMI: {:.2f}%".format(100 * score))

    print("RandIndex: {:.2f}%".format(100 * randIndex(y_train, pre_train)))
