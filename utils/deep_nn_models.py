
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import  AveragePooling2D, Conv2DTranspose
from keras.layers import Concatenate, Dropout, Cropping2D
from keras.regularizers import l2

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Activation
import numpy as np


# Horat & Lerch 2023

class Unet:

    def __init__(self, v, train_patches=False, weighted_loss=False,
                 ct_kernel=(3, 3), ct_stride=(2, 2), n_blocks=3, filters=2,
                 apool=True, bn=True):

        self.train_patches = train_patches
        self.model_architecture = 'unet'
        self.weighted_loss = weighted_loss

        # params related to input/preproc.
        if self.train_patches == False:
            self.input_dims = 0
            self.output_dims = 0
        else:
            self.input_dims = 32
            self.output_dims = self.input_dims 
            self.patch_stride = 12
            self.patch_na = 4 / 8
        self.n_bins = 3
        self.region = 'global'  # 'europe'

        # params for model architecture
        self.filters = filters #2
        self.apool = apool  # True choose between average and max pooling, True = average
        self.n_blocks = n_blocks  # 4  # 5
        self.bn = bn  # True batch normalization
        self.ct_kernel = ct_kernel #(3, 3)  # (2, 2)
        self.ct_stride = ct_stride #(2, 2)  # (2, 2)

        # params related to model training
        self.optimizer_str = 'adam'
        self.call_back = True  # should early stopping be used?

        if v == 'tp':
            self.learn_rate = 0.001
            self.decay_rate = 0.005
            self.delayed_early_stop = True
        else:
            self.learn_rate = 1e-4
            self.decay_rate = 0
            self.delayed_early_stop = False

        if self.train_patches == True:
            self.bs = 32
            self.ep = 20
            self.patience = 3  # for callback
            self.start_epoch = 2  # epoch to start with early stopping
        else:  # global unet
            self.bs = 16
            self.ep = 50  # 20
            self.patience = 10  # for callback
            self.start_epoch = 5
            if self.call_back == False:
                self.ep = 30

    def build_model(self,  dg_train_shape, dg_train_weight_target=None,output="proba"):
        #modfication here 
        inp_imgs = Input(shape=(dg_train_shape[0],
                                dg_train_shape[1],
                                dg_train_shape[2]))  # fcts

        c0 = inp_imgs

        # encoder / contracting path
        p1, c1 = down(c0, self.filters*4, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 16
        p2, c2 = down(p1, self.filters*8, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 8
        p3, c3 = down(p2, self.filters*16, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 4
        p4, c4 = down(p3, self.filters*32, activation='elu', padding='same',  bn=self.bn, apool=self.apool) if (self.n_blocks >= 4) else [p3, c3]
        p5, c5 = down(p4, self.filters*64, activation='elu', padding='same',  bn=self.bn, apool=self.apool) if (self.n_blocks >= 5) else [p4, c4]

        # bottleneck
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same', name='bottleneck')(p5)
        # cb = Dropout(self.dropout_rate)(cb)
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same')(cb)
        cb = BatchNormalization()(cb) if self.bn else cb

        # decoder / expanding path
        u5 = up(cb, c5, self.filters*64, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn) if (self.n_blocks >=5 ) else cb
        u4 = up(u5, c4, self.filters*32, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn) if (self.n_blocks >=4 ) else cb
        u3 = up(u4, c3, self.filters*16, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn)
        u2 = up(u3, c2, self.filters*8, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn)
        u1 = up(u2, c1, self.filters*4, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=False)  # no normalization directly before softmax

        # output layer
        if output == "proba":
            out = Conv2D(3, (1, 1), activation='softmax')(u1)
        elif output == "deterministic":
            out = Conv2D(1, (1, 1), activation='relu')(u1)


        # crop to get rid of patch edges
        if self.train_patches == True:
            out = out #Cropping2D(cropping=((4, 4), (4, 4)))(out)
            #out = Cropping2D(cropping=((3, 3), (1, 1)))(out)
        else:
            if self.region == 'europe':
                out = Cropping2D(cropping=((8, 8), (8, 8)))(out)
            if self.region == 'global':
                #out = Cropping2D(cropping=((3, 3), (1, 1)))(out)
                out = out # Cropping2D(cropping=((8, 8), (4, 3)))(out)

        if (self.train_patches == True) & (self.weighted_loss == True):
            weight_shape = dg_train_weight_target[0]
            weights = Input(shape=(weight_shape[1], weight_shape[2],))
            target_shape = dg_train_weight_target[1]
            target = Input(shape=(target_shape[1], target_shape[2], target_shape[3],))
            inputs = [inp_imgs]

            cnn = Model(inputs=[inputs] + [weights, target], outputs=out)

            cnn.target = target
            cnn.weight_mask = weights
            cnn.out = out
        else:
            cnn = Model(inputs=[inp_imgs], outputs=out)

        # cnn.summary()

        return cnn


def down(c, filters, activation='elu', padding='same', lamda=0,
         dropout_rate=0, bn=True, apool=True):
    # lamda: l2 regularizer for kernel and bias
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)
    c = Dropout(dropout_rate)(c)
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)
    c = BatchNormalization()(c) if bn else c
    p = AveragePooling2D((2, 2))(c) if apool else MaxPooling2D((2, 2))(c)
    return p, c


def up(u, c, filters, ct_kernel, ct_stride, activation='elu',
       padding='same', lamda=0, dropout_rate=0, bn=True):
    u = Conv2DTranspose(filters, ct_kernel, strides=ct_stride, padding=padding,
                        kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # 8x8
    u = Concatenate()([c, u])  # 8x8
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # pad = same 8x8
    u = Dropout(dropout_rate)(u)
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # pad = same 8x8
    u = BatchNormalization()(u) if bn else u
    return u


def MLP(input_shape, num_classes=3):
    #input shpe must be (lat,lon)
    model = Sequential()
    # Flatten the input
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(2048, kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())  
    model.add(Dropout(0.3))  
    
    model.add(Dense(512, kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output Layer with softmax
    model.add(Dense(np.prod(input_shape) * num_classes, activation='linear')) 
    model.add(Reshape(input_shape + (num_classes,)))  # Reshape to (, , 3)
   #softmax on last dim
    model.add(Activation('softmax'))

    return model

def CNN(input_shape=(32, 32, 1), num_filters=16, output_channels=1):
    inputs = Input(shape=input_shape, name='input')

    # First Convolutional Block
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

    # Upsample to restore spatial dimensions
    x = Conv2D(num_filters * 2, kernel_size=(3, 3), activation='relu', padding='same')(x)

    x = Conv2D(num_filters*4, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Output layer: Convolutional layer to produce (lat, lon, output_channels)
    x = Conv2D(output_channels, kernel_size=(3, 3), activation='softmax', padding='same')(x)

    model = Model(inputs, x)
    return model
