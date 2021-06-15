# 128
import numpy as np
import os
from keras import backend as K
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, UpSampling2D
from keras.layers import Conv2D, ConvLSTM2D, BatchNormalization, Concatenate, AveragePooling2D, TimeDistributed
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
import timeit


img_size  = 128
seq_frame = 30
l2        = regularizers.l2(0.0001)
bias_init = keras.initializers.glorot_uniform()


def dice(y_true,
         y_pred):

    smooth       = 1e-12

    y_pred       = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    intersection = K.sum(y_true * y_pred)
    union        = K.sum(y_true) + K.sum(y_pred) - intersection

    return (2. * intersection + smooth) / (union + intersection + smooth)


def seg_loss(y_true,
             y_pred):

    smooth       = 1e-12

    y_true_back  = 1 - y_true
    y_pred_back  = 1 - y_pred

    alpha        = 1 / (K.pow(K.sum(y_true), 2) + smooth)
    beta         = 1 / (K.pow(K.sum(y_true_back), 2) + smooth)

    numerater    = alpha * K.sum(y_true * y_pred) + beta * K.sum(y_true_back * y_pred_back)
    denominator  = alpha * K.sum(y_true + y_pred) + beta * K.sum(y_true_back + y_pred_back)

    dice_loss    = 1 - (2. * numerater) / (denominator + smooth)
    mae_loss     = K.mean(K.log(1 + K.exp(K.abs(y_pred - y_true))))

    w            = (img_size * img_size - K.sum(y_pred)) / (K.sum(y_pred) + smooth)
    key_w        = 0.003

    crossentropy = - K.mean(key_w * w * y_true * K.log(y_pred + smooth) + y_true_back * K.log(y_pred_back + smooth))

    return crossentropy + dice_loss + mae_loss


def conv2d(tensor,
           filters,
           kernel_size =3,
           dilated_rate=1,
           bias=False):

    if bias is True:
        tensor = TimeDistributed(Conv2D(filters, kernel_size, dilation_rate=dilated_rate, padding='same',
                                        bias_initializer=bias_init,
                                        kernel_regularizer=l2)
                                 )(tensor)
    else:
        tensor = TimeDistributed(Conv2D(filters, kernel_size, dilation_rate=dilated_rate, padding='same',
                                        use_bias=False,
                                        kernel_regularizer=l2)
                                 )(tensor)
    return tensor


def bac(tensor,
        filters,
        kernel_size,
        dilated_rate,
        drop):

    tensor = TimeDistributed(BatchNormalization())(tensor)
    tensor = TimeDistributed(Activation('relu'))(tensor)
    tensor = conv2d(tensor, filters, kernel_size, dilated_rate)
    tensor = TimeDistributed(Dropout(rate=drop))(tensor)
    return tensor


def block(tensor,
          block_layers,
          growth,
          dilated_rate,
          drop):

    for i in range(block_layers):
        tmp    = bac(tensor, growth, 3, dilated_rate, drop)
        tensor = Concatenate(axis=-1)([tensor, tmp])

    channels = np.int32(tensor.shape[-1])

    return tensor, channels


def ral(input_shape=(seq_frame, img_size, img_size, 1),
        layers =None,
        growth =None,
        dropout=None,
        fuse_ch=None,
        theta  =None):

    inputs     = Input(shape=input_shape)  # (batch size, (30, 128, 128, 1))

    # l1 128*128
    tensor     = conv2d(inputs, filters=16)
    tensor, ch = block(tensor, layers, growth, dilated_rate=1, drop=dropout)
    l1_128     = conv2d(tensor, filters=fuse_ch, bias=True)
    tensor     = bac(tensor, filters=np.int32(ch * theta), kernel_size=1, dilated_rate=1, drop=dropout)
    tensor     = TimeDistributed(AveragePooling2D((2, 2), padding='valid'))(tensor)

    # l2 64*64
    tensor, ch = block(tensor, layers, growth, dilated_rate=1, drop=dropout)
    l2_64      = conv2d(tensor, filters=fuse_ch, bias=True)
    tensor     = bac(tensor, filters=np.int32(ch * theta), kernel_size=1, dilated_rate=1, drop=dropout)
    tensor     = TimeDistributed(AveragePooling2D((2, 2), padding='valid'))(tensor)

    # l3 32*32
    tensor, ch = block(tensor, layers, growth, dilated_rate=2, drop=dropout)
    l3_32      = conv2d(tensor, filters=fuse_ch, bias=True)
    tensor     = bac(tensor, filters=np.int32(ch * theta), kernel_size=1, dilated_rate=1, drop=dropout)

    # l4 32*32
    tensor, ch = block(tensor, layers, growth, dilated_rate=4, drop=dropout)
    l4_32      = conv2d(tensor, filters=fuse_ch, bias=True)
    tensor     = bac(tensor, filters=np.int32(ch * theta), kernel_size=1, dilated_rate=1, drop=dropout)

    # l5 32*32
    tensor, ch = block(tensor, layers, growth, dilated_rate=8, drop=dropout)
    l5_32      = conv2d(tensor, filters=fuse_ch, bias=True)

    # r5 32*32
    tensor     = ConvLSTM2D(filters=fuse_ch, kernel_size=3, padding='same', return_sequences=True)(l5_32)
    tensor     = TimeDistributed(BatchNormalization())(tensor)
    r5_32      = conv2d(tensor, 1, 3)

    # r4 32*32
    tensor     = Concatenate(axis=-1)([l4_32, tensor])
    tensor     = ConvLSTM2D(filters=fuse_ch, kernel_size=3, padding='same', return_sequences=True)(tensor)
    tensor     = TimeDistributed(BatchNormalization())(tensor)
    r4_32      = conv2d(tensor, 1, 3)

    # r3 32*32
    tensor     = Concatenate(axis=-1)([l3_32, tensor])
    tensor     = ConvLSTM2D(filters=fuse_ch, kernel_size=3, padding='same', return_sequences=True)(tensor)
    tensor     = TimeDistributed(BatchNormalization())(tensor)
    r3_32      = conv2d(tensor, 1, 3)

    # r2 64*64
    tensor     = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(tensor)
    tensor     = Concatenate(axis=-1)([l2_64, tensor])
    tensor     = ConvLSTM2D(filters=fuse_ch, kernel_size=3, padding='same', return_sequences=True)(tensor)
    tensor     = TimeDistributed(BatchNormalization())(tensor)
    r2_64      = conv2d(tensor, 1, 3, bias=True)

    # r1 128*128
    tensor     = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(tensor)
    tensor     = Concatenate(axis=-1)([l1_128, tensor])
    tensor     = ConvLSTM2D(filters=fuse_ch, kernel_size=3, padding='same', return_sequences=True)(tensor)
    tensor     = TimeDistributed(BatchNormalization())(tensor)
    r1_128     = conv2d(tensor, 1, 3, bias=True)

    # fuse and putout segmentation
    r5_to_128  = TimeDistributed(UpSampling2D((4, 4), interpolation='bilinear'))(r5_32)
    r4_to_128  = TimeDistributed(UpSampling2D((4, 4), interpolation='bilinear'))(r4_32)
    r3_to_128  = TimeDistributed(UpSampling2D((4, 4), interpolation='bilinear'))(r3_32)
    r2_to_128  = TimeDistributed(UpSampling2D((2, 2), interpolation='bilinear'))(r2_64)
    yp         = Concatenate(axis=-1)([r5_to_128,
                                       r4_to_128,
                                       r3_to_128,
                                       r2_to_128,
                                       r1_128])
    yp         = conv2d(yp, 1, 3, bias=True)
    yp         = Activation('sigmoid', name='yp')(yp)

    # classification
    l5_to_32   = conv2d(l5_32, 1, 3)
    r5_to_32   = r5_32
    r4_to_32   = r4_32
    r3_to_32   = r3_32
    r2_to_16   = TimeDistributed(AveragePooling2D((4, 4) , padding='valid'))(r2_64)
    r1_to_8   = TimeDistributed(AveragePooling2D((16, 16), padding='valid'))(r1_128)

    pool5      = Concatenate(axis=-1)([l5_to_32, r5_to_32])
    pool5      = conv2d(pool5, 1, 3, bias=True)

    pool4      = Concatenate(axis=-1)([pool5   , r4_to_32])
    pool4      = conv2d(pool4, 1, 3, bias=True)

    pool3      = Concatenate(axis=-1)([pool4   , r3_to_32])
    pool3      = conv2d(pool3, 1, 3, bias=True)
    pool3      = TimeDistributed(AveragePooling2D((2, 2)  , padding='valid'))(pool3)

    pool2      = Concatenate(axis=-1)([pool3   , r2_to_16])
    pool2      = conv2d(pool2, 1, 3, bias=True)
    pool2      = TimeDistributed(AveragePooling2D((2, 2)  , padding='valid'))(pool2)

    pool1      = Concatenate(axis=-1)([pool2   , r1_to_8])
    pool1      = conv2d(pool1, 1, 3, bias=True)
    pool1      = TimeDistributed(AveragePooling2D((2, 2)  , padding='valid'))(pool1)

    pool1      = TimeDistributed(Flatten())(pool1)
    pool1      = TimeDistributed(Dense(256, activation='relu'))(pool1)
    pool1      = TimeDistributed(Dense(64 , activation='relu'))(pool1)
    classify   = Dense(3  , activation='softmax', name='cl')(pool1)

    ral_model  = Model(inputs=inputs, outputs=[yp, classify])

    return ral_model


if __name__ == '__main__':
    bd_a2c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/ims_a2c.npy')[:, :, :, :, np.newaxis]
    bd_a3c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/ims_a3c.npy')[:, :, :, :, np.newaxis]
    bd_a4c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/ims_a4c.npy')[:, :, :, :, np.newaxis]
    bd_a2c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/gts_a2c.npy')[:, :, :, :, np.newaxis]
    bd_a3c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/gts_a3c.npy')[:, :, :, :, np.newaxis]
    bd_a4c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/bd_ge_vivide9/gts_a4c.npy')[:, :, :, :, np.newaxis]

    hk1_a2c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/ims_a2c.npy')[:, :, :, :, np.newaxis]
    hk1_a3c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/ims_a3c.npy')[:, :, :, :, np.newaxis]
    hk1_a4c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/ims_a4c.npy')[:, :, :, :, np.newaxis]
    hk1_a2c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/gts_a2c.npy')[:, :, :, :, np.newaxis]
    hk1_a3c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/gts_a3c.npy')[:, :, :, :, np.newaxis]
    hk1_a4c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_epiq7c/gts_a4c.npy')[:, :, :, :, np.newaxis]

    hk2_a2c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/ims_a2c.npy')[:, :, :, :, np.newaxis]
    hk2_a3c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/ims_a3c.npy')[:, :, :, :, np.newaxis]
    hk2_a4c_ims = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/ims_a4c.npy')[:, :, :, :, np.newaxis]
    hk2_a2c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/gts_a2c.npy')[:, :, :, :, np.newaxis]
    hk2_a3c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/gts_a3c.npy')[:, :, :, :, np.newaxis]
    hk2_a4c_gts = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/hk_philips_ie33/gts_a4c.npy')[:, :, :, :, np.newaxis]

    sz_a2c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/ims_a2c.npy')[:, :, :, :, np.newaxis]
    sz_a3c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/ims_a3c.npy')[:, :, :, :, np.newaxis]
    sz_a4c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/ims_a4c.npy')[:, :, :, :, np.newaxis]
    sz_a2c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/gts_a2c.npy')[:, :, :, :, np.newaxis]
    sz_a3c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/gts_a3c.npy')[:, :, :, :, np.newaxis]
    sz_a4c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/gts_a4c.npy')[:, :, :, :, np.newaxis]

    a2c_ims     = np.concatenate((bd_a2c_ims, hk1_a2c_ims, hk2_a2c_ims, sz_a2c_ims), axis=0)
    a3c_ims     = np.concatenate((bd_a3c_ims, hk1_a3c_ims, hk2_a3c_ims, sz_a3c_ims), axis=0)
    a4c_ims     = np.concatenate((bd_a4c_ims, hk1_a4c_ims, hk2_a4c_ims, sz_a4c_ims), axis=0)
    a2c_gts     = np.concatenate((bd_a2c_gts, hk1_a2c_gts, hk2_a2c_gts, sz_a2c_gts), axis=0)
    a3c_gts     = np.concatenate((bd_a3c_gts, hk1_a3c_gts, hk2_a3c_gts, sz_a3c_gts), axis=0)
    a4c_gts     = np.concatenate((bd_a4c_gts, hk1_a4c_gts, hk2_a4c_gts, sz_a4c_gts), axis=0)
    print(a2c_ims.shape, a2c_gts.shape)
    print(a3c_ims.shape, a3c_gts.shape)
    print(a4c_ims.shape, a4c_gts.shape)
    print('\n')

    a2c_lb      = np.repeat(np.array([[1., 0., 0.]], dtype=np.float32), seq_frame, axis=0)
    a3c_lb      = np.repeat(np.array([[0., 1., 0.]], dtype=np.float32), seq_frame, axis=0)
    a4c_lb      = np.repeat(np.array([[0., 0., 1.]], dtype=np.float32), seq_frame, axis=0)
    print(a2c_lb.shape, a3c_lb.shape, a4c_lb.shape)
    print('\n')

    a2c_lbs     = np.repeat([a2c_lb], a2c_ims.shape[0], axis=0)
    a3c_lbs     = np.repeat([a3c_lb], a3c_ims.shape[0], axis=0)
    a4c_lbs     = np.repeat([a4c_lb], a4c_ims.shape[0], axis=0)
    print(a2c_lbs.shape, a3c_lbs.shape, a4c_lbs.shape)
    print('\n')

    ims         = np.concatenate((a2c_ims, a3c_ims, a4c_ims), axis=0)
    gts         = np.concatenate((a2c_gts, a3c_gts, a4c_gts), axis=0)
    lbs         = np.concatenate((a2c_lbs, a3c_lbs, a4c_lbs), axis=0)
    print(ims.shape, gts.shape, lbs.shape)
    print('\n')

    # random
    np.random.seed(9)
    pi          = np.random.permutation(ims.shape[0])
    ims         = ims[pi]
    gts         = gts[pi]
    lbs         = lbs[pi]

    train_ims   = ims[:270, :, :, :, :]
    train_gts   = gts[:270, :, :, :, :]
    train_lbs   = lbs[:270, :]

    val_ims     = ims[270:300, :, :, :, :]
    val_gts     = gts[270:300, :, :, :, :]
    val_lbs     = lbs[270:300, :]

    test_ims    = ims[300:, :, :, :, :]
    test_gts    = gts[300:, :, :, :, :]
    test_lbs    = lbs[300:, :]

    print(train_ims.shape, train_gts.shape, train_lbs.shape)
    print(val_ims.shape  , val_gts.shape  , val_lbs.shape )
    print(test_ims.shape , test_gts.shape , test_lbs.shape )
    print('\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    batch_size  = 1
    epochs      = 150    #100

    model       = ral(input_shape=(30, img_size, img_size, 1),
                      layers =6,
                      growth =16,
                      dropout=0.25,        #0.2
                      fuse_ch=20,
                      theta  =0.5)

    model.summary()

    plot_model(model,
               to_file='/home/lm/Desktop/model.png',
               show_shapes=True)

    opt         = Adam(lr=0.0015)     #0.001

    model.compile(optimizer   =opt,
                  loss        ={'yp': seg_loss,
                                'cl': 'categorical_crossentropy'},
                  loss_weights={'yp': 1.0,
                                'cl': 0.3},
                  metrics     ={'yp': [dice],
                                'cl': 'accuracy'})

    es          = EarlyStopping(monitor ='val_yp_dice',
                                patience=10,
                                mode    ='max',
                                verbose =1)
    ckpt        = ModelCheckpoint('/home/sshsz/ls/data/file_to_lishuang/data_miccai/weights/ral.hdf5_V1',
                                  save_best_only=True,
                                  monitor       ='val_yp_dice',
                                  mode          ='max',
                                  verbose       =1)
    reduce_lr   = ReduceLROnPlateau(monitor ='val_yp_dice',
                                    factor  =0.8,
                                    patience=5,
                                    verbose =1,
                                    mode    ='max',
                                    min_lr  =1e-8)#1e-9

    model.fit(x              =train_ims,
              y              =[train_gts, train_lbs],
              batch_size     =batch_size,
              epochs         =epochs,
              shuffle        =False,
              validation_data=[val_ims, [val_gts, val_lbs]],
              callbacks      =[es, ckpt, reduce_lr],
              verbose        =1)

    score = model.evaluate(x         =test_ims,
                           y         =[test_gts, test_lbs],
                           batch_size=batch_size)
    print(score)
