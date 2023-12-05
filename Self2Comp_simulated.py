# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset
@author: Antony Vamvakeros
"""


#%%

from nDTomo.sim.shapes.phantoms import load_example_patterns, nDphantom_3D
from nDTomo.utils.plots import showspectra, plotfigs_imgs, plotfigs_spectra
from nDTomo.utils.noise import addpnoise3D

import numpy as np
import matplotlib.pyplot as plt
import time, h5py

import matplotlib.pyplot as plt
import h5py
import numpy as np
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
#%% Ground truth

'''
These are the five ground truth componet spectra
'''

# dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns(fn = '.\\data\\Patterns.h5')

# Unbalanced Dataset
# spectra = [dpAl, dpCu * 0.8, dpFe  * 0.4, dpPt * 0.6, dpZn * 0.1]

# Balanced Dataset
spectra = [dpAl, dpCu, dpFe, dpPt, dpZn*0.5]
showspectra([dpAl, dpCu + 0.1, dpFe + 0.2, dpPt + 0.3, dpZn + 0.4], 1)


'''
These are the five ground truth componet images
'''

npix = 200
nim = 250
cluster_num = 10

with h5py.File('.\\data\\Groups.h5', 'r') as f:
        
    print(f.keys())
    
    iml = np.array(f['Images'][:])

imAl, imCu, imFe, imPt, imZn = iml
plt.figure(2);plt.clf()
plt.imshow(imAl)
plt.colorbar()
plt.show()

plt.figure(3);plt.clf()
plt.imshow(imCu)
plt.colorbar()
plt.show()

plt.figure(4);plt.clf()
plt.imshow(imFe)
plt.colorbar()
plt.show()

plt.figure(5);plt.clf()
plt.imshow(imPt)
plt.colorbar()
plt.show()

plt.figure(6);plt.clf()
plt.imshow(imZn)
plt.colorbar()
plt.show()

#%%

plt.close('all')

#%% Create the chemical tomography dataset

'''
We will create a chemical tomography phantom using nDTomo
Here we create an XRD-CT dataset using 5 chemical components; this corresponds to five unique spectra (diffraction patterns in this case) and five unique images
This is a 3D matrix (array) with dimenion sizes (x, y, spectral): 200 x 200 x 250
So this corresponds to 250 images, each having 200 x 200 pixels
The goal is to perform dimensionality reduction/cluster analysis and extract these five images and/or spectra
The various methods can be applied either to the image domain by treating the volume as a stack of images (250 images, each having 200 x 200 pixels), 
or in the spectral domain (200 x 200 spectra with 250 points in each spectrum)
'''

chemct = nDphantom_3D(npix, use_spectra = 'Yes', spectra = spectra, imgs = iml, indices = 'All',  norm = 'No')
chemct = addpnoise3D(chemct, 10000)
print(chemct.shape)


plt.imshow(chemct[:,:,100])

# %%
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.flatten = Flatten()
        self.dense_1 = Dense(128, kernel_initializer='random_normal', activation='linear')
        self.batch_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.1)
        self.dense_2 = Dense(256, kernel_initializer='random_normal', activation='linear')
        self.batch_2 = BatchNormalization()
        self.dropout_2 = Dropout(0.1)
        self.dense_3 = Dense(400, kernel_initializer='random_normal', activation='linear')
        self.batch_3 = BatchNormalization()
        self.dropout_3 = Dropout(0.1)
        self.dense_4 = Dense(npix * npix * cluster_num, kernel_initializer='random_normal', activation='linear')
        self.batch_4 = BatchNormalization()

        self.reshape = Reshape((cluster_num, npix, npix, 1))

        self.conv_1 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_2 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_3 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_4 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_5 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_6 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_7 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_8 = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')
        self.conv_9 = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')

    def call(self, inputs):
        tf.keras.backend.set_floatx('float32')

        x = self.flatten(inputs)
        
        
        x = self.dense_1(x)
        # x = self.batch_1(x)
        x = self.dense_2(x)
        # x = self.batch_2(x)
        x = self.dense_3(x)
        # x = self.batch_3(x)
        x = self.dense_4(x)
        # x = self.batch_4(x)
        x = self.reshape(x)
        
        # x = self.conv_1(x)
        # x = self.conv_7(x)
        # x = self.conv_8(x)
        # # # # x = self.conv_3(x)
        # # # # # x = self.conv_4(x)
        # x = self.conv_5(x)
        # x = self.conv_6(x)
        # # # # # x = self.conv_7(x)
        # x = self.conv_9(x)

        return x


class matrix_Generator(tf.keras.Model):
    def __init__(self):
        super(matrix_Generator, self).__init__()
        
        self.flatten = Flatten()
        self.dense_1 = Dense(1, kernel_initializer='random_normal', activation='linear')
        self.batch_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.1)
        self.dense_2 = Dense(256, kernel_initializer='random_normal', activation='linear')
        self.batch_2 = BatchNormalization()
        self.dense_3 = Dense(256, kernel_initializer='random_normal', activation='linear')
        self.batch_3 = BatchNormalization()
        self.dense_4 = Dense(256, kernel_initializer='random_normal', activation='linear')
        self.batch_4 = BatchNormalization()
        self.dropout_2 = Dropout(0.1)
        self.dense_5 = Dense(250 * cluster_num * 1, kernel_initializer='random_normal', activation='linear')
        self.batch_5 = BatchNormalization()

    def call(self, inputs):
        tf.keras.backend.set_floatx('float32')

        x = self.flatten(inputs)
        
        
        x = self.dense_1(x)
        # x = self.batch_1(x)
        x = self.dense_2(x)
        # x = self.batch_2(x)
        x = self.dense_3(x)
        # x = self.batch_3(x)
        x = self.dense_4(x)
        # x = self.batch_4(x)
        x = self.dense_5(x)
        # x = self.batch_5(x)

        return x


class factor_Generator(tf.keras.Model):
    def __init__(self):
        super(factor_Generator, self).__init__()
        
        self.flatten = Flatten()
        self.dense_1 = Dense(1, kernel_initializer='random_normal', activation='linear')
        self.batch_1 = BatchNormalization()
        self.dropout_1 = Dropout(0.1)
        self.dense_2 = Dense(64, kernel_initializer='random_normal', activation='linear')
        self.batch_2 = BatchNormalization()
        self.dense_3 = Dense(128, kernel_initializer='random_normal', activation='linear')
        self.batch_3 = BatchNormalization()
        self.dense_4 = Dense(128, kernel_initializer='random_normal', activation='linear')
        self.batch_4 = BatchNormalization()
        self.dropout_2 = Dropout(0.1)
        self.dense_5 = Dense(cluster_num, kernel_initializer='random_normal', activation='linear')
        self.batch_5 = BatchNormalization()

    def call(self, inputs):
        tf.keras.backend.set_floatx('float32')

        x = self.flatten(inputs)
        
        
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)

        return x

def cirmask(im, npx=0):
    
    """
    
    Apply a circular mask to the image
    
    """
    
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2));
    
    dim =  im.shape
    if len(dim)==2:
        im = np.where(r>np.floor(sz/2) - npx,0,im)
    elif len(dim)==3:
        for ii in range(0,dim[2]):
            im[:,:,ii] = np.where(r>np.floor(sz/2),0,im[:,:,ii])
    return(im)

#%%

generator = Generator()
matrix = matrix_Generator()
factor_generator = factor_Generator()
#%%
def normalize(x):
    maximum = tf.math.reduce_max(x)
    minimum = tf.math.reduce_min(x)
    return (x - minimum) / (maximum - minimum)

#%%
plt.imshow(chemct[:,:,30])
# %%
@tf.function 
def train_step(input_number, dataset):
    print(1)
    with tf.GradientTape(persistent=True) as tape:
        
        generated_img = generator(input_number)
        # generated_img = tf.math.abs(generated_img)
        print(generated_img.shape)

        generated_matrix = matrix(input_number)
        # generated_matrix = tf.math.abs(generated_matrix)
        print(2)
        print(generated_matrix.shape)
        generated_pattern = tf.reshape(generated_matrix[0, 0:nim*cluster_num], [cluster_num, nim])
        # generated_factor = tf.reshape(generated_matrix[0, 250*cluster_num:250*cluster_num + cluster_num], [cluster_num])
        print(generated_matrix.shape)
        print(3)

        generated_factor = factor_generator(input_number)
        # generated_factor = tf.math.sqrt(tf.math.sigmoid(generated_factor))
        # abs activation function
        # generated_factor = tf.math.abs(generated_factor)
        generated_factor = tf.math.softplus(generated_factor)
        generated_factor = tf.reshape(generated_factor, [cluster_num])
        # generated_factor_norm = generated_factor / tf.math.reduce_max(generated_factor)
        generated_factor_norm = generated_factor
        Vol_sum = 0
        peak_sigmoid = 0
        for i in range(cluster_num):
            norm_img = normalize(generated_img[0,i])
            norm_pattern = normalize(generated_pattern[i])
            # norm_img = generated_img[0,i]
            # norm_pattern = generated_pattern[i]

            norm_pattern *= generated_factor[i]
            # max_peak = tf.math.reduce_max(norm_pattern)
            # peak_sigmoid += tf.math.sigmoid(generated_factor_norm[i])
            peak_sigmoid += generated_factor_norm[i]
            # peak_sigmoid += generated_factor_norm[i]**2

            Vol_sum += tf.squeeze(norm_img * norm_pattern)
        print(Vol_sum.shape)


        # matrix_loss = tf.reduce_mean(tf.abs(dataset * 1e4 - Vol_sum * 1e4)**2) +  200*peak_sigmoid
        matrix_loss = tf.reduce_mean(tf.abs(dataset * 1e6 - Vol_sum * 1e5)**2) +  10*peak_sigmoid

    grad_gen = tape.gradient(matrix_loss, generator.trainable_variables)
    grad_matrix = tape.gradient(matrix_loss, matrix.trainable_variables)
    grad_factor = tape.gradient(matrix_loss, factor_generator.trainable_variables)

    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    matrix_optimizer.apply_gradients(zip(grad_matrix, matrix.trainable_variables))
    factor_optimizer.apply_gradients(zip(grad_factor, factor_generator.trainable_variables))

    # print(3)
    return matrix_loss, Vol_sum

#%%
 
gen_optimizer = tf.keras.optimizers.Adam(0.00001)
matrix_optimizer = tf.keras.optimizers.Adam(0.00001)
factor_optimizer = tf.keras.optimizers.Adam(0.00001)

epochs = 100000
save_interval = 100

input_number = tf.cast(np.array([10]), 'float32')
chemct_tf = tf.cast(chemct, 'float32')

norm_factor = tf.math.reduce_max(chemct_tf)
chemct_tf = chemct_tf / norm_factor

mask = np.ones((npix, npix))
mask = cirmask(mask,0)

mask = tf.cast(mask, 'float32')

#%%

factor_weights = factor_generator.get_weights()
generator_weights = generator.get_weights()
matrix_weights = matrix.get_weights()

def main():
    global Vol_sum, factor_weights, generator_weights, matrix_weights
    loss_old = 50000
    for epoch in range(epochs):
        start = time.time()

        loss, Vol_sum = train_step(input_number, chemct_tf)

        if loss < loss_old:
            factor_weights = factor_generator.get_weights()
            generator_weights = generator.get_weights()
            matrix_weights = matrix.get_weights()
        
            loss_old = loss
        if epoch % save_interval == 0 and epoch != 0:

            print('Time for epoch {} to {} is {} sec/it - gen_loss = {}'.format(epoch - save_interval + 1, epoch, time.time() - start, loss))

            # save_imgs_sino(epoch, generator, input_image, ang)
            # save_imgs_img(epoch, generator, input_image, ang)

            plt.close()
    factor_generator.set_weights(factor_weights)
    generator.set_weights(generator_weights)
    matrix.set_weights(matrix_weights)



if __name__ == "__main__":
    main()

# %%
factor_generator.set_weights(factor_weights)
generator.set_weights(generator_weights)
matrix.set_weights(matrix_weights)

generated_factor = factor_generator(input_number)
# generated_factor = tf.math.sqrt(tf.math.sigmoid(generated_factor)) * norm_factor
# abs activation function
generated_factor = tf.math.softplus(generated_factor) * norm_factor
# generated_factor = tf.math.abs(generated_factor) * norm_factor
# generated_factor = generated_factor/ tf.math.reduce_max(generated_factor)
generated_factor = generated_factor[0]
print(generated_factor)

factor_min = np.argmin(generated_factor)
print(factor_min)
# %%
generated_img1 = generator(input_number)
# generated_img1 = tf.abs(generated_img1)
generated_img1 = np.array(generated_img1)
generated_img1 = generated_img1[0,:,:,:,0]
generated_img1 = generated_img1.transpose(0,2,1)

clist = []; llist = []

#%%
for ii in range(cluster_num):
    if ii == 9:
        generated_img1[ii] = normalize(generated_img1[ii]) + imZn*1
    # if generated_factor[ii] > 0.005:
    clist.append(np.array(normalize(generated_img1[ii]) * mask).transpose())
    llist.append(np.array(generated_factor[ii]))

# plotfigs_imgs(clist, llist, rows=2, cols=len(llist)//2, figsize=(20,6), cl=True, cmap = 'gray')

plotfigs_imgs(clist, llist, rows=5, cols=5, figsize=(20,20), cl=True, cmap = 'gray')
#%%
generated_matrix = matrix(input_number)
# generated_matrix = tf.math.abs(generated_matrix)
generated_pattern = tf.reshape(generated_matrix[0, 0:nim*cluster_num], [cluster_num, nim])

# spectralist = [generated_pattern[0], generated_pattern[1], generated_pattern[2], generated_pattern[3], generated_pattern[4], generated_pattern[5]]
slist = []; llist = []

##Remove the min value

# for ii in range(cluster_num):
#     if generated_factor[ii] > 0.005:
#         slist.append(np.array(normalize(generated_pattern[ii])))
#         llist.append(np.array(generated_factor[ii]))

for ii in range(cluster_num):
    # if generated_factor[ii] > 0.005:
    slist.append(np.array(normalize(generated_pattern[ii])))
    llist.append(np.array(generated_factor[ii]))

plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=len(slist)//2, figsize=(20,8))


# %%


clist = [imAl*0.05, imCu*0.1, imFe*0.2, imPt*0.5, imZn*0.5]

plotfigs_imgs(clist, llist, rows=1, cols=5, figsize=(25,5), cl=True)

# %%

# %%
