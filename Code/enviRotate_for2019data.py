# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:
    Rotate ENVI files.
    Make sure to change the following matrices accordingly.
    1) original_shift;
    2) rotate_mtx;
    3) translation_vec

Version: 1.0

Reference:
"""



'''Set working directory and all files' paths'''
# =============================================================================
# import os
# work_dir = "C:/GoogleDrive/code"
# os.chdir(work_dir)  
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import spectral.io.envi as envi
from scipy.interpolate import griddata
import copy 
start_time = time.time()


"""===========================MAIN PROGRAM BELOW============================"""
time_stamps = ['08051158', '08121245', '08141222', '08161204', '08201154']
tstp = time_stamps[4]
img_path = rf'F:\2019snapbeans\SfM\snb_{tstp}\snb_{tstp}\4_index\reflectance\enviOutput\{tstp}_bgrren_VIs_mask.dat'
img_hdr_path = img_path[:-4]+'.hdr'
img = envi.open(img_hdr_path, img_path)

print(img.__class__)
print(img)
print(img.shape)

original_shift = [-335400.00, -4744000.00, 0.00] #make sure these to be the same as in CloudCompare.
map_info = img.metadata['map info']
x_start = float(map_info[3]) + original_shift[0]
y_start = float(map_info[4]) + original_shift[1]
pix_size = float(map_info[5])

#%%
coords_x = np.arange(img.shape[1])
coords_y = np.arange(img.shape[0])
yp, xp = np.meshgrid(coords_y, coords_x, indexing='ij')
pix_xy = np.array([xp.ravel()* pix_size, yp.ravel()* -pix_size]).T + np.array([x_start, y_start]) #n*2 array
#%%
#1. transform the coordinates of the pixels and attach them as extra bands
rotate_mtx = np.array([[0.998776197433, -0.049458608031],
                     [0.049458608031, 0.998776197433]])
translation_vec = np.array([4.509527206421, 0.754966080189])
trans_pix_xy = (rotate_mtx @ pix_xy.T).T + translation_vec 


#%%
#2. do bilinear interpolation on the extended image with coordinates.
grid_min = np.min(trans_pix_xy, axis=0)
grid_max = np.max(trans_pix_xy, axis=0)
grid_x, grid_y = np.mgrid[grid_min[0]:grid_max[0]:pix_size, 
                          grid_max[1]:grid_min[1]:-pix_size] #notice that y decreases from top to bottom.

img_arr = np.asarray(img.load())
rotate_img = []
intp_method = 'nearest'
for i in range(img_arr.shape[2]):
    single_band = np.clip(img_arr[:, :, i], a_min=0, a_max=None)
    # single_band[single_band==-10000]=0
    single_band_pix = single_band.ravel()
    
    band_intp = griddata(trans_pix_xy, single_band_pix, (grid_x, grid_y), method=intp_method, fill_value=0).T
    band_intp = np.clip(band_intp, a_min=0, a_max=None)
        
    rotate_img.append(band_intp)
    
    #Plot the transform results
    # org_extent = (pix_xy[:,0].min(), pix_xy[:,0].max(), pix_xy[:,1].min(), pix_xy[:,1].max())
    # plt_extent = (grid_min[0], grid_max[0], grid_min[1], grid_max[1])
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(single_band, extent=org_extent, origin='upper')
    # plt.title('Original')
    # plt.subplot(222)
    # plt.imshow(grid_z0, extent=plt_extent, origin='upper')
    # plt.title('Nearest')
    # plt.subplot(223)
    # plt.imshow(band_intp, extent=plt_extent, origin='upper')
    # plt.title('Linear')
    # plt.subplot(224)
    # plt.imshow(grid_z2, extent=plt_extent, origin='upper')
    # plt.title('Cubic')
    # plt.gcf().set_size_inches(6, 6)
    # plt.show()

    print(f'Band {i+1} finished!!')


#%%
#save the file
metadata = copy.deepcopy(img.metadata)
metadata['lines'] = str(band_intp.shape[0])
metadata['samples'] = str(band_intp.shape[1])
metadata['map info'][3] = str(grid_min[0] - original_shift[0]) #xmin
metadata['map info'][4] = str(grid_max[1] - original_shift[1]) #ymax


metadata['description'] = 'rotated in python with interpolation method - nearest neighbor'
save_img = np.stack(rotate_img, axis=2) #stack the bands, in 'bip' order - (Rows, Cols, Bands)
save_img_hdr_path = img_hdr_path[:-4]+'_rotate_nn.hdr'
envi.save_image(save_img_hdr_path, save_img, metadata=metadata, force=True)
#%%
print("--- %.1f seconds ---" % (time.time() - start_time))