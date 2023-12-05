#%%
from nDTomo.sim.shapes.phantoms import load_example_patterns, nDphantom_3D, nDphantom_4D, nDphantom_2Dmap

from nDTomo.utils.plots import closefigs, showspectra, showim, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.h5data import h5write_data, h5read_data
from nDTomo.utils.noise import addpnoise3D


import numpy as np
import matplotlib.pyplot as plt
import time, h5py

### Packages for clustering and dimensionality reduction

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
# from clustimage import Clustimage
import pymcr

#%%

'''
Part 1: Data generation
'''
def plotfigs_imgs(imagelist, legendlist=None, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'gray'):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(imagelist)):

            legendlist.append('Component %d' %(ii+1))        
        
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
            
                print(kk)
                
                if kk < len(imagelist):
            
                    i = axes[ii,jj].imshow(imagelist[kk], cmap=cmap)
                    axes[ii,jj].set_axis_off()
                    axes[ii,jj].set_title(legendlist[kk])
            
                    if cl==True:
                        fig.colorbar(i, ax=axes[ii,jj])        
                    
                    kk = kk + 1

#% Ground truth

'''
These are the five ground truth componet spectra
'''

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns(fn = 'F:\\Dropbox (Finden)\\Finden team folder\\AI\\Segmentation\\nDTomo\\nDTomo\\examples\\patterns\\Patterns.h5')
# Unbalanced dataset
# spectra = [dpAl, dpCu * 0.8, dpFe  * 0.6, dpPt * 0.4, dpZn * 0.1]

# Balanced dataset
spectra = [dpAl, dpCu, dpFe, dpPt, dpZn*0.5]
showspectra([dpAl, dpCu + 0.1, dpFe + 0.2, dpPt + 0.3, dpZn + 0.4], 1)
spa = np.array(spectra)

'''
These are the five ground truth componet images
'''

npix = 200
# This creates a list containing five images, all with the same dimensions
# iml = nDphantom_2D(npix, nim = 'Multiple')
# print(len(iml))

with h5py.File('F:\\Dropbox (Finden)\\Finden team folder\\AI\\Segmentation\\nDTomo\\nDTomo\\examples\\patterns\\Groups.h5', 'r') as f:
        
    print(f.keys())
    
    iml = np.array(f['Images'][:])

imAl, imCu, imFe, imPt, imZn = iml

showim(imAl, 2)
showim(imCu, 3)
showim(imFe, 4)
showim(imPt, 5)
showim(imZn, 6)

#%% Ground truth data

gtimlist = [imAl, imCu, imFe, imPt, imZn]
gtsplist = [dpAl, dpCu, dpFe, dpPt, dpZn]
gtldlist = ['Al', 'Cu', 'Fe', 'Pt', 'Zn']


#%% Close the various figures

closefigs()

#%% Let's create a 3D (chemical-CT) dataset with two spatial and one spectral dimensions (x,y,spectral)

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

# Add noise in to the chemct
chemct = addpnoise3D(chemct, 10000)
print('The volume dimensions are %d, %d, %d' %(chemct.shape[0], chemct.shape[1], chemct.shape[2]))

#%% Finally let's create a 2D chemical map after taking a projection from a 3D chemical-CT dataset

vol4d = nDphantom_4D(npix = 200, nzt = 100, vtype = 'Spectral', indices = 'Random', spectra=spectra, imgs=iml, norm = 'Volume')

print('The volume dimensions are %d, %d, %d, %d' %(vol4d.shape[0], vol4d.shape[1], vol4d.shape[2], vol4d.shape[3]))

#%% Now create a projection dataset from the 3D chemical-ct dataset

map2D = nDphantom_2Dmap(vol4d, dim = 0)

print('The map dimensions are %d, %d, %d' %(map2D.shape[0], map2D.shape[1], map2D.shape[2]))


# %%
#%% Export the simulated data
newpath = './data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

p = 'data\\'
fn = 'phantom_data'

h5write_data(p, fn, ['ground_truth_images', 'ground_truth_spectra', 'tomo', 'map'], [gtimlist, gtsplist, chemct, map2D])

#%%

'''
Part 2: Data analysis
'''

#%% Let's load the data

# Specify the path to the data
fn = './data/phantom_data.h5'

data = h5read_data(fn,  ['ground_truth_images', 'ground_truth_spectra', 'tomo', 'sinograms', 'map'])
print(data.keys())

#%%

gtimlist = data['ground_truth_images']
gtsplist = data['ground_truth_spectra']
chemct = data['tomo']
map2D = data['map']
# %%
#%% PCA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

imagelist, legendlist = create_complist_imgs(pca.components_, chemct.shape[0], chemct.shape[1])

imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii].transpose())
    llist.append(legendlist[ii])
for ii in range(len(imagelist)):
    imagelist[ii][imagelist[ii]<0] = 0
    clist.append(imagelist[ii])
    llist.append(legendlist[ii])


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

#%% K means: images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

imagelist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    imagelist.append(np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2))


imagelist = [imagelist[2], imagelist[4], imagelist[3], imagelist[0], imagelist[1]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii].transpose())
    llist.append(legendlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))



plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

#%% AgglomerativeClustering: images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 6

ac = AgglomerativeClustering(distance_threshold=None, linkage="complete", n_clusters=clusters).fit(data)

labels = ac.labels_[:]

imagelist = []; inds = []
for ii in range(np.max(labels)):

    inds.append(np.where(ac.labels_==ii))
    
    tmp = chemct[:,:,np.squeeze(inds[ii])]
    
    if len(tmp.shape)>2:
    
        imagelist.append(np.mean(tmp, axis = 2))
        
    else:
        
        imagelist.append(tmp)

    print(ii)
    
imagelist = [imagelist[4], imagelist[1], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

#%% Spectral clustering: images

spc = SpectralClustering(affinity="nearest_neighbors", n_clusters=5, eigen_solver="arpack").fit(data)

labels = spc.labels_[:]

imagelist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(spc.labels_==ii))
    
    imagelist.append(np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2))

# imagelist = [imagelist[3], imagelist[0], imagelist[1], imagelist[4], imagelist[2]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

#%% NMF: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

start = time.time()
nmf = NMF(n_components=5, max_iter = 5000).fit(data+0.01)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

imagelist, legendlist = create_complist_imgs(nmf.components_, chemct.shape[0], chemct.shape[1])

imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Component %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

# %%
'''
Clustering on the spectra domain
'''
#%% PCA: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

spectralist, legendlist = create_complist_spectra(pca.components_)

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Component %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


#%% K means: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


#%% AgglomerativeClustering: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

ac = AgglomerativeClustering(n_clusters=clusters).fit(data)

labels = ac.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(ac.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))

#%% SpectralClustering: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

spc = SpectralClustering(affinity="nearest_neighbors", n_clusters=5, eigen_solver="arpack").fit(data)

labels = spc.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(ac.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))

#%% NMF: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

nmf = NMF(n_components=6).fit(data+0.01)
print(nmf.components_.shape)

spectralist, legendlist = create_complist_spectra(nmf.components_)

spectralist = [spectralist[1], spectralist[4], spectralist[2], spectralist[3], spectralist[0]]

#%%
slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    spect = spectralist[ii]
    spect = spect / spect.max()
    slist.append(spect)
    llist.append('Component %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


# %%
