from model import *

bd_a2c_ims  = np.load('/home/sshsz/ls/data/data_miccai/ims_a2c.npy')[:, :,:, :, np.newaxis]
bd_a3c_ims  = np.load('/home/sshsz/ls/data/data_miccai/ims_a3c.npy')[:, :, :, :, np.newaxis]
bd_a4c_ims  = np.load('/home/sshsz/ls/data/data_miccai/ims_a4c.npy')[:, :, :, :, np.newaxis]
bd_a2c_gts  = np.load('/home/sshsz/ls/data/data_miccai/gts_a2c.npy')[:, :, :, :, np.newaxis]
bd_a3c_gts  = np.load('/home/sshsz/ls/data/data_miccai/gts_a3c.npy')[:, :, :, :, np.newaxis]
bd_a4c_gts  = np.load('/home/sshsz/ls/data/data_miccai/gts_a4c.npy')[:, :, :, :, np.newaxis]
bd_a2c_ims  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/ims_a2c.npy')[:, :, :, :, np.newaxis]
bd_a3c_ims  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/ims_a3c.npy')[:, :, :, :, np.newaxis]
bd_a4c_ims  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/ims_a4c.npy')[:, :, :, :, np.newaxis]
bd_a2c_gts  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/gts_a2c.npy')[:, :, :, :, np.newaxis]
bd_a3c_gts  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/gts_a3c.npy')[:, :, :, :, np.newaxis]
bd_a4c_gts  = np.load('/home/amax/projectls1/data_miccai/bd_ge_vivide9/gts_a4c.npy')[:, :, :, :, np.newaxis]

hk1_a2c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/ims_a2c.npy')[:, :, :, :, np.newaxis]
hk1_a3c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/ims_a3c.npy')[:, :, :, :, np.newaxis]
hk1_a4c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/ims_a4c.npy')[:, :, :, :, np.newaxis]
hk1_a2c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/gts_a2c.npy')[:, :, :, :, np.newaxis]
hk1_a3c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/gts_a3c.npy')[:, :, :, :, np.newaxis]
hk1_a4c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_epiq7c/gts_a4c.npy')[:, :, :, :, np.newaxis]

hk2_a2c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/ims_a2c.npy')[:, :, :, :, np.newaxis]
hk2_a3c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/ims_a3c.npy')[:, :, :, :, np.newaxis]
hk2_a4c_ims = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/ims_a4c.npy')[:, :, :, :, np.newaxis]
hk2_a2c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/gts_a2c.npy')[:, :, :, :, np.newaxis]
hk2_a3c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/gts_a3c.npy')[:, :, :, :, np.newaxis]
hk2_a4c_gts = np.load('/home/amax/projectls1/data_miccai/hk_philips_ie33/gts_a4c.npy')[:, :, :, :, np.newaxis]

sz_a2c_ims  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/ims_a2c.npy')[ :,:, :, :,np.newaxis]
sz_a3c_ims  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/ims_a3c.npy')[ :,:, :, :,np.newaxis]
sz_a4c_ims  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/ims_a4c.npy')[ :,:, :, :,np.newaxis]
sz_a2c_gts  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/gts_a2c.npy')[ :,:, :, :,np.newaxis]
sz_a3c_gts  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/gts_a3c.npy')[ :,:, :, :,np.newaxis]
sz_a4c_gts  = np.load('/home/amax/projectls1/data_miccai/sz_philips_epiq7c/gts_a4c.npy')[ :,:, :, :,np.newaxis]

a2c_ims     = np.concatenate((bd_a2c_ims, hk1_a2c_ims, hk2_a2c_ims, sz_a2c_ims), axis=0)
a3c_ims     = np.concatenate((bd_a3c_ims, hk1_a3c_ims, hk2_a3c_ims, sz_a3c_ims), axis=0)
a4c_ims     = np.concatenate((bd_a4c_ims, hk1_a4c_ims, hk2_a4c_ims, sz_a4c_ims), axis=0)
a2c_gts     = np.concatenate((bd_a2c_gts, hk1_a2c_gts, hk2_a2c_gts, sz_a2c_gts), axis=0)
a3c_gts     = np.concatenate((bd_a3c_gts, hk1_a3c_gts, hk2_a3c_gts, sz_a3c_gts), axis=0)
a4c_gts     = np.concatenate((bd_a4c_gts, hk1_a4c_gts, hk2_a4c_gts, sz_a4c_gts), axis=0)

# a2c_ims     = np.concatenate((bd_a2c_ims), axis=0)
# a3c_ims     = np.concatenate((bd_a3c_ims), axis=0)
# a4c_ims     = np.concatenate((bd_a4c_ims), axis=0)
# a2c_gts     = np.concatenate((bd_a2c_gts), axis=0)
# a3c_gts     = np.concatenate((bd_a3c_gts), axis=0)
# a4c_gts     = np.concatenate((bd_a4c_gts), axis=0)
print(a2c_ims.shape, a2c_gts.shape)
print(a3c_ims.shape, a3c_gts.shape)
print(a4c_ims.shape, a4c_gts.shape)
print('\n')

# a2c_ims     = bd_a2c_ims
# # a3c_ims     = bd_a3c_ims
# a4c_ims     = bd_a4c_ims
# a2c_gts     = bd_a2c_gts
# # a3c_gts     = bd_a3c_gts
# a4c_gts     = bd_a4c_gts
# na_a2c_ims = np.load('F:/na/test/na_a4c/ims_a2c.npy')[:, :, :, :, np.newaxis]
# ca_a3c_ims  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/ims_a3c.npy')[:, :, :, :, np.newaxis]
# na_a4c_ims = np.load('F:/na/test/na_a4c/ims_a4c.npy')[:, :, :, :, np.newaxis]
# na_a2c_gts = np.load('F:/na/test/na_a4c/gts_a2c.npy')[:, :, :, :, np.newaxis]
# ca_a3c_gts  = np.load('/home/sshsz/ls/data/file_to_lishuang/data_miccai/sz_philips_epiq7c/gts_a3c.npy')[:, :, :, :, np.newaxis]
# na_a4c_gts = np.load('F:/na/test/na_a4c/gts_a4c.npy')[:, :, :, :, np.newaxis]
# ca2c_ims = np.repeat(ca_a2c_ims, 15, axis=1)
# ca4c_ims = np.repeat(ca_a4c_ims, 15, axis=1)
# ca2c_gts = np.repeat(ca_a2c_gts, 15, axis=1)
# ca4c_gts = np.repeat(ca_a4c_gts, 15, axis=1)

# a2c_ims = na2c_ims
# a4c_ims = na_a4c_ims
# a2c_gts = ca2c_gts
# a4c_gts = na_a4c_gts
# print(a2c_ims.shape, a2c_gts.shape) #(50, 2, 1, 256, 256) (50, 0, 1)
# # print(a3c_ims.shape, a3c_gts.shape)
# print(a4c_ims.shape, a4c_gts.shape) #(50, 2, 1, 256, 256) (50, 0, 1)
# print('\n')

a2c_lb      = np.repeat(np.array([[1., 0., 0.]], dtype=np.float32), seq_frame, axis=0)
a3c_lb      = np.repeat(np.array([[0., 1., 0.]], dtype=np.float32), seq_frame, axis=0)
a4c_lb      = np.repeat(np.array([[0., 0., 1.]], dtype=np.float32), seq_frame, axis=0)
print(a4c_lb.shape) #(30, 3) (30, 3)
print('\n')
a2c_lbs     = np.repeat([a2c_lb], a2c_ims.shape[0], axis=0)
a3c_lbs     = np.repeat([a3c_lb], a3c_ims.shape[0], axis=0)
a4c_lbs     = np.repeat([a4c_lb], a4c_ims.shape[0], axis=0)
print(a2c_lbs.shape,  a4c_lbs.shape) #(50, 30, 3) (50, 30, 3)
print('\n')

ims         = np.concatenate((a2c_ims, a3c_ims, a4c_ims), axis=0)
gts         = np.concatenate((a2c_gts, a3c_gts, a4c_gts), axis=0)
lbs         = np.concatenate((a2c_lbs, a3c_lbs, a4c_lbs), axis=0)
ims         = np.concatenate(( a4c_ims), axis=0)
gts         = np.concatenate((a4c_gts), axis=0)
lbs         = np.concatenate((a4c_lbs), axis=0)
# ims         = a4c_ims
# gts         = a4c_gts
# lbs         = a4c_lbs

print(ims.shape, gts.shape, lbs.shape) #(100, 2, 1, 256, 256) (100, 0, 1) (100, 30, 3)
print('\n')

# random
np.random.seed(9)
pi          = np.random.permutation(ims.shape[0])
ims         = ims[pi]
gts         = gts[pi]
lbs         = lbs[pi]

train_ims   = ims[:270, :, :]
train_gts   = gts[:270, :, :]
train_lbs   = lbs[:270, :]

val_ims     = ims[270:300, :, :, :, :]
val_gts     = gts[270:300, :, :, :, :]
val_lbs     = lbs[270:300, :]

test_ims    = ims[:, :,  :]
test_gts    = gts[:, :,  :]
test_lbs    = lbs[:, :]

print(train_ims.shape, train_gts.shape, train_lbs.shape)
print(val_ims.shape  , val_gts.shape  , val_lbs.shape )
print(test_ims.shape , test_gts.shape , test_lbs.shape ) #(100, 2, 1, 256, 256) (100, 0, 1) (100, 30, 3)
print('\n')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size  = 10
epochs      = 100

model       = ral(input_shape=(30, img_size, img_size, 1),
                  layers =6,
                  growth =16, #16
                  dropout=0.2, #0.2
                  fuse_ch=20,
                  theta  =0.5) #0.5

model.load_weights('F:/ls/ral.hdf5_N4')
model.summary()
from keras.utils import plot_model
plot_model(model,to_file='./model.')
opt = Adam(lr=0.001)

model.compile(optimizer   =opt,
              loss        ={'yp': seg_loss,
                            'cl': 'categorical_crossentropy'},
              loss_weights={'yp': 1.0,   #1.0
                            'cl': 0.3},  #0.3
              metrics     ={'yp': [dice],
                            'cl': 'accuracy'})

score = model.evaluate(x         =test_ims,
                       y         =[test_gts, test_lbs],
                        batch_size=batch_size)
print(score)
