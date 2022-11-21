# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:

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
import pandas as pd
from copy import deepcopy
start_time = time.time()


"""===========================MAIN PROGRAM BELOW============================"""
time_stamps = ['08051158', '08121245', '08141222', '08161204', '08201154']
tstp = time_stamps[4]
img_path = rf'F:\2019snapbeans\SfM\snb_{tstp}\snb_{tstp}\4_index\reflectance\enviOutput\{tstp}_bgrren_VIs_mask_rotate_nn.img'
img_hdr_path = img_path[:-4]+'.hdr'
img = envi.open(img_hdr_path, img_path)

print(img.__class__)
print(img)
print(img.shape)

#%%
original_shift = [-335400.00, -4744000.00] #make sure these to be the same as in CloudCompare.
map_info = img.metadata['map info']
x_start = float(map_info[3]) + original_shift[0]
y_start = float(map_info[4]) + original_shift[1]
pix_size = float(map_info[5])

coords_x = np.arange(img.shape[1])
coords_y = np.arange(img.shape[0])
yp, xp = np.meshgrid(coords_y, coords_x, indexing='ij')
pix_xy = np.array([xp.ravel()* pix_size, yp.ravel()* -pix_size]).T + np.array([x_start, y_start]) #n*2 array

#%%
#Read the boundaries from the txt file.
import pickle

bound_file = r'F:\2019snapbeans\lidar\2019reprocessed\20190805\1158\forYieldPaper\08051158_res_i_for_seg_clean_bound.pkl'
with open(bound_file, 'rb') as bfr:
    pic = pickle.loads(bfr.read())

# print(pic)
y_limits_ls = pic['y_limits_ls'] + original_shift[1]
x_limits_ls = pic['x_limits_ls'] + original_shift[0]




#%%
#assgin the row labels by using the boudaries.
seg_labels = np.zeros(img.shape[0]*img.shape[1], dtype=np.uint8)
# for i, y_lim in enumerate(y_limits):
#     if i%2==0:
#         down_limit = y_lim
#         up_limit = y_limits[i+1]
#         obs_r_pts_mask = np.logical_and(pix_xy[:, 1]<up_limit, pix_xy[:, 1]>down_limit)
#     print(f'Row {i}.')
#     for j,v_bound in enumerate(v_sumZ_bounds[i//2]):
#         if j%2==0:
#             left_limit = v_bound
#             right_limit = v_sumZ_bounds[i//2][j+1]
#             obs_c_pts_mask = np.logical_and(pix_xy[:, 0]<right_limit, pix_xy[:, 0]>left_limit)
#             seg_mask = np.logical_and(obs_r_pts_mask, obs_c_pts_mask)
#             print(f'Number of seg points: {np.sum(seg_mask)}')
#             seg_labels[seg_mask] = np.intc(i//2*3 + j//2)
i=1
for col_x_lim, col_y_lim in zip(x_limits_ls, y_limits_ls):
    for row_x_lim, row_y_lim in zip(col_x_lim, col_y_lim):
        down_limit = row_y_lim[0]
        up_limit = row_y_lim[1]
        left_limit = row_x_lim[0]
        right_limit = row_x_lim[1]
        seg_mask = np.logical_and.reduce((pix_xy[:, 1]<up_limit, 
                                        pix_xy[:, 1]>down_limit, 
                                        pix_xy[:, 0]>left_limit, 
                                        pix_xy[:, 0]<right_limit))
        # print(f'Row {i}.')
        seg_labels[seg_mask] = i
        i+=1

#%%
#save the segmentation labels to an envi file.
metadata = deepcopy(img.metadata)
metadata['description'] = 'Row segments labels'
metadata['bands'] = '1'
metadata['band names'] = ['row label']

save_img_rowseg = seg_labels.reshape(img.shape[:2]) #stack the bands
save_img_hdr_path_rowseg = img_hdr_path[:-4]+'_rowseg.hdr'
# img_save_rowseg = envi.create_image(save_img_hdr_path_rowseg, metadata)
# mm_rowseg = img_save_rowseg.open_memmap(writable=True)
# mm_rowseg[:, :, 0] = save_img_rowseg[:]
# print(img_save_rowseg.__dict__)
envi.save_image(save_img_hdr_path_rowseg, save_img_rowseg, metadata=metadata, force=True)


#%%
def calcVIRpr(envi_img, seg_labels, row_label):
    '''
    Calculate either mean of median of the VIs for each row segment.
    '''
    envi_img_pix = np.asarray(envi_img.load()).reshape((-1, img.shape[2]))
    row_pix = envi_img_pix[seg_labels==row_label] #n_pix * n_bands
    print(f'num of row_pix = {len(row_pix)}')
    row_veg_pix = row_pix[row_pix[:, -1]==1]
    print(f'num of row_veg_pix = {len(row_veg_pix)}')
    # if method=='ave':
    #     rpr_VI = np.average(row_veg_pix, axis=0)
    # elif method=='med':
    #     rpr_VI = np.median(row_veg_pix, axis=0)
        
    rpr_VI = np.average(row_veg_pix, axis=0)

    
    return rpr_VI[5:-1] #first five bands were b/g/r/re/nir, and then followed by VIs
    

#%%
#calculate the representatives within each row segment and then save the results.
seg_index = [j for j in range(1, seg_labels.max()+1)]
ave_VIs, med_VIs = [], []
for si in seg_index:
    print(f'Row {si}')
    ave_VI = calcVIRpr(img, seg_labels, si)
    ave_VIs.append(ave_VI)

    
#%%
sav_df1 = pd.DataFrame(np.stack(ave_VIs, axis=0), 
                      columns=img.metadata['band names'][5:-1])   
sav_df1.insert(0, 'seg_index', seg_index)
sav_df1.to_csv(img_path[:-4]+'_row_ave.csv', index=False) 

#%%
print("--- %.1f seconds ---" % (time.time() - start_time))