# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset
@author: Antony Vamvakeros
"""


#%%

import numpy as np
import matplotlib.pyplot as plt
import time, h5py

### Packages for clustering and dimensionality reduction

from sklearn.decomposition import NMF
#%%
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

cluster_num = 10

with h5py.File('.\\data\\D20.h5', 'r') as f:
        
    print(f.keys())
    
    iml = np.array(f['images'][:])
    tth = np.array(f['twotheta'])

iml = iml[100:670]
tth = tth[100:670]
plt.imshow(iml[0])

npix = iml.shape[1]
nim = iml.shape[0]
#%%
plt.plot(tth, iml.sum(axis=(1,2)))
#%%

plt.close('all')

#%% chemical tomography dataset

chemct = iml.transpose([1,2,0])
print(chemct.shape)
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
        self.dense_5 = Dense(nim * cluster_num * 1, kernel_initializer='random_normal', activation='linear')
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

def plotfigs_imgs(imagelist, legendlist, epoch, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'jet'):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(imagelist)):
            
            i = axes[ii].imshow(imagelist[ii], cmap=cmap)
            axes[ii].set_axis_off()
            axes[ii].set_title(legendlist[ii])
    
            if cl==True:
                fig.colorbar(i, ax=axes[ii])

    elif len(axes.shape)==2:
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                # print(kk)
                
                if kk < len(imagelist):
            
                    i = axes[ii,jj].imshow(imagelist[kk], cmap=cmap)
                    axes[ii,jj].set_axis_off()
                    axes[ii,jj].set_title(legendlist[kk])
            
                    if cl==True:
                        fig.colorbar(i, ax=axes[ii,jj])        
                    
                    kk = kk + 1
    fig.savefig('result/D20_epoch{}.png'.format(epoch))
    plt.close()
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
        generated_factor = tf.math.abs(generated_factor)
        # generated_factor = tf.math.softplus(generated_factor)
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
            peak_sigmoid += tf.math.sigmoid(generated_factor_norm[i])
            # peak_sigmoid += generated_factor_norm[i]
            # peak_sigmoid += generated_factor_norm[i]**2

            # maximum_pattern = tf.math.reduce_max(generated_matrix[i])
            # minimum_pattern = tf.math.reduce_min(generated_matrix[i])
            # norm_pattern = (generated_matrix[i] - minimum_pattern) / (maximum_pattern - minimum_pattern)
            # print(norm_pattern)
            Vol_sum += tf.squeeze(norm_img * norm_pattern)
        print(Vol_sum.shape)


        matrix_loss = tf.reduce_mean(tf.abs(dataset * 5e4 - Vol_sum * 5e4)**2) +  50*peak_sigmoid
        # matrix_loss = tf.reduce_mean(tf.abs(dataset * 1e4 - Vol_sum * 1e4)**2) +  1*peak_sigmoid

    grad_gen = tape.gradient(matrix_loss, generator.trainable_variables)
    grad_matrix = tape.gradient(matrix_loss, matrix.trainable_variables)
    grad_factor = tape.gradient(matrix_loss, factor_generator.trainable_variables)

    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    matrix_optimizer.apply_gradients(zip(grad_matrix, matrix.trainable_variables))
    factor_optimizer.apply_gradients(zip(grad_factor, factor_generator.trainable_variables))

    # print(3)
    return matrix_loss, Vol_sum

#%%
 
gen_optimizer = tf.keras.optimizers.Adam(0.0001)
matrix_optimizer = tf.keras.optimizers.Adam(0.0001)
factor_optimizer = tf.keras.optimizers.Adam(0.0001)

epochs = 30000
save_interval = 100

input_number = tf.cast(np.array([1]), 'float32')
chemct_tf = tf.cast(chemct, 'float32')

norm_factor = tf.math.reduce_max(chemct_tf)
chemct_tf = chemct_tf / norm_factor

mask = np.ones((npix, npix))
mask = cirmask(mask,0)

mask = tf.cast(mask, 'float32')

#%%
epochs = 50000
factor_weights = factor_generator.get_weights()
generator_weights = generator.get_weights()
matrix_weights = matrix.get_weights()

def main():
    global Vol_sum, factor_weights, generator_weights, matrix_weights
    loss_old = 1000000
    count = 0
    start = time.time()
    for epoch in range(epochs):

        loss, Vol_sum = train_step(input_number, chemct_tf)


        
        if epoch % save_interval == 0 and epoch != 0:

            print('Time for epoch {} to {} is {} sec/it - gen_loss = {}'.format(epoch - save_interval + 1, epoch, time.time() - start, loss))

            #manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
            # save_imgs_sino(epoch, generator, input_image, ang)
            # save_imgs_img(epoch, generator, input_image, ang)

            plt.close()

        # if (epoch - 4) % 1000 == 0 and epoch != 0:

        #     factor_generator.save_weights('factor_generator_{}_4'.format(count))
        #     generator.save_weights('generator_{}_4'.format(count))
        #     matrix.save_weights('matrix_{}_4'.format(count))
        #     print('image saved, count:', count)
            
        #     #Plot the figure
        #     generated_factor = factor_generator(input_number)
        #     generated_factor = tf.math.abs(generated_factor) * norm_factor
        #     generated_factor = generated_factor[0]

        #     factor_min = np.argmin(generated_factor)
        #     generated_img1 = generator(input_number)
        #     generated_img1 = np.array(generated_img1)
        #     generated_img1 = generated_img1[0,:,:,:,0]
        #     generated_img1 = generated_img1.transpose(0,2,1)

        #     clist = []; llist = []
        #     for ii in range(cluster_num):
        #         clist.append(np.array(normalize(generated_img1[ii]) * mask).transpose())
        #         llist.append(np.array(generated_factor[ii]))

        #     plotfigs_imgs(clist, llist, epoch, rows=5, cols=5, figsize=(20,20), cl=True, cmap = 'gray')

        #     count += 1
        #     # if count == 10:
        #     #     count = 0

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
# generated_factor = tf.math.softplus(generated_factor) * norm_factor
generated_factor = tf.math.abs(generated_factor) * norm_factor
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
# imagelist = [generated_img1[0], generated_img1[1], generated_img1[2], generated_img1[3], generated_img1[4], generated_img1[5], generated_img1[6], generated_img1[7]]

clist = []; llist = []

for ii in range(cluster_num):
    # if ii == 3:
    #     generated_img1[ii] = normalize(generated_img1[ii]) + imZn*1
    # if generated_factor[ii] > 0.005:+
    clist.append(np.array(normalize(generated_img1[ii]) * mask).transpose())
    llist.append(np.array(generated_factor[ii]))

# plotfigs_imgs(clist, llist, rows=2, cols=len(llist)//2, figsize=(20,6), cl=True, cmap = 'gray')

plotfigs_imgs(clist, llist, rows=5, cols=5, figsize=(20,20), cl=True, cmap = 'gray')
#%%
def plotfigs_spectra(spectralist, legendlist= None, epoch=0, xaxis=None, rows=1, cols=5, figsize=(20,3)):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(legendlist)):

            legendlist.append('Component %d' %(ii+1))    
    
    if xaxis is None:
        
        xaxis = np.arange(0, len(spectralist[0]))
            
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(spectralist)):
            
            axes[ii].plot(xaxis, spectralist[ii])
            axes[ii].set_title(legendlist[ii])

    elif len(axes.shape)==2:

        print(1)
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                print(kk)
                
                if kk < len(spectralist):
            
                    axes[ii,jj].plot(tth, spectralist[kk])
                    axes[ii,jj].set_title('Component ' + str(kk + 1) + ' Strength: ' + format(legendlist[kk], '.3f'),fontsize = 15)
                    axes[ii,jj].set_xlabel('2θ (°)',fontsize = 15)

   
                    kk = kk + 1
    fig.savefig('result/spectra_epoch{}.png'.format(epoch))

generated_matrix = matrix(input_number)
# generated_matrix = tf.math.abs(generated_matrix)
generated_pattern = tf.reshape(generated_matrix[0, 0:nim*cluster_num], [cluster_num, nim])

# spectralist = [generated_pattern[0], generated_pattern[1], generated_pattern[2], generated_pattern[3], generated_pattern[4], generated_pattern[5]]
slist = []; llist = []

##Remove the min value

for ii in range(cluster_num):
    # if generated_factor[ii] > 0.005:
    slist.append(np.array(normalize(generated_pattern[ii])))
    llist.append(np.array(generated_factor[ii]))

plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=len(slist)//2, figsize=(20,8))


# %%
llist = np.array(llist)
clist = np.array(clist)
slist = np.array(slist)
with h5py.File('F:\\Dropbox (Finden)\\Finden team folder\\AI\\Segmentation\\D20_seg_{}_2.h5'.format(cluster_num), 'w') as hf:
    #with h5py.File('F:\\Dropbox (Finden)\\Finden team folder\\AI\\Parallax\\Libraries\\Parallax\\Recon_Images_%d.h5'%(interval), 'w') as hf:
    hf.create_dataset("factor",  data=generated_factor)
    hf.create_dataset("image",  data=clist)
    hf.create_dataset("pattern",  data=llist)


# %%
from sklearn.decomposition import NMF
data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()
#%%
print(data.shape)

def create_complist_imgs(components, xpix, ypix):     

    imagelist = []; legendlist = []
    for ii in range(components.shape[0]):
        im = components[ii,:]
        imagelist.append(np.reshape(im, (xpix, ypix)))
        legendlist.append('Component %d' %(ii+1))
        
    return(imagelist, legendlist)

start = time.time()
nmf = NMF(n_components=10, max_iter = 5000).fit(data+0.01)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

imagelist, legendlist = create_complist_imgs(nmf.components_, chemct.shape[0], chemct.shape[1])

# imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

#%%
clist = []; llist = []

for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Component %d' %(ii + 1))

#%%
plotfigs_imgs(clist, llist, epoch='NMF10', rows=2, cols=5, figsize=(20,13), cl=True, cmap = 'gray')

#%%
with h5py.File(r'D20_seg_20.h5', 'r') as f:
        
    print(f.keys())
    
    generated_img1 = np.array(f['image'][:])
    generated_factor = np.array(f['factor'])
    generated_pattern = np.array(f['pattern'])

#%%

clist = []; llist = []

for ii in range(cluster_num):
    # if ii == 3:
    #     generated_img1[ii] = normalize(generated_img1[ii]) + imZn*1
    # if generated_factor[ii] > 0.005:+
    clist.append(np.array(normalize(generated_img1[ii]) * mask))
    llist.append(np.array(generated_factor[ii]))

# plotfigs_imgs(clist, llist, rows=2, cols=len(llist)//2, figsize=(20,6), cl=True, cmap = 'gray')

plotfigs_imgs(clist, llist, epoch='Self2Comp_10', rows=5, cols=4, figsize=(20,23), cl=True, cmap = 'gray')
