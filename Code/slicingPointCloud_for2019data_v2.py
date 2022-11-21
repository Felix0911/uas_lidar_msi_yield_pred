# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 00:12:32 2022

Description: This code segments the plot of 2019 snap beans data into 
row segments:
    1. Keep only the above ground points, which were extracted by using a 
        hight threshold on z-residual from the rotated and markers-removed 
        point cloud.
    2. Separate the points into columns.
    3. For each column, separate each rows and find the up and bottom edges 
        for each row.
    4. For each row, find the left and right edges.

@author: Fei Zhang
"""

#set working directory to the script's own directory
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%
import numpy as np
import laspy
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from CalcRprstPerSlice import CalcRprstPerSlice
import copy
from findVerticalBounds_SingleColumn_for2019 import findConsecutiveOnes
import pickle
#%%
def slicePointCloud(slice_width, las_pts, slice_axis=1):
    '''slice_axis: 0-cut across x-axis; 1-cut across y-axis
    '''
    len_slice = las_pts[:,slice_axis].max() - las_pts[:,slice_axis].min()
    slice_idx_max = int(len_slice // slice_width)
    sliced_pts = []
    for slice_idx in range(slice_idx_max):
        # slice_idx = 0
        slice_bot = las_pts[:,slice_axis].min() + slice_idx * slice_width
        slice_top = slice_bot + slice_width
        mask = np.logical_and(las_pts[:,slice_axis]>slice_bot, las_pts[:,slice_axis]<slice_top) 
        slice_pts = las_pts[mask]
        sliced_pts.append(slice_pts)
    
    return sliced_pts


def displayPointCloud(pts_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)    
    o3d.visualization.draw_geometries([pcd])
    return True

    

def plotMapWithMaxMinHorizontal(py, m_idx, label):
    '''Finding Maxima or Minima'''
    plt.figure()
    # for py, m_idx, label in zip(pys, m_idxs, labels):
    px = np.linspace(0, las_y.max(), len(py))
    # plt.plot(px, py, '.-', mfc='none', label=label)
    py_norm = np.asarray(py)/max(py)
    plt.plot(px, py_norm, linestyle='--', label=label)
    plt.scatter(px[m_idx], np.zeros(len(m_idx)), s=300, marker="*", alpha=0.5, label=label+' border')
    plt.xlabel('x') # unit - m
    plt.ylabel('Cell metric') # (normalized to 0-1
    # for m_i in m_idx:
    #     plt.axvline(x=px[m_i], c='r', linestyle='--')
    plt.grid()
    plt.legend(loc=1)
    plt.show()


def findRowOnes(in_arr, thresh_num=1, verbose=True):
    #step 1
    idx_ls = []
    cont_ones_idx = []
    for i, ele in enumerate(in_arr):
        if ele==1 and (i==0 or in_arr[i-1]==0):
            cont_ones_idx.append(i)
        if ele==1 and (i==len(in_arr)-1 or in_arr[i+1]==0):
            cont_ones_idx.append(i)
            idx_ls.append(cont_ones_idx)
            cont_ones_idx = []
        else:
            continue

    if verbose:   
        print(f'The bounds of the consecutive ones are \n {idx_ls}.')
    

    #step 2
    new_idx_ls = copy.deepcopy(idx_ls)
    cnt = 0 
    i=0
    if verbose: 
        print(f'Combining the consecutive ones within gap zeros of {thresh_num}.')
    while i<len(idx_ls)-1:
        if len(new_idx_ls)==1:
            break
        dist = new_idx_ls[i+1-cnt][0] - new_idx_ls[i-cnt][1]
        if dist < thresh_num:
            new_idx = [new_idx_ls[i-cnt][0], new_idx_ls[i-cnt+1][1]]
            new_idx_ls.pop(i-cnt)
            new_idx_ls[i-cnt] = new_idx
            if verbose:   
                print(f'i={i}, new_idx_ls={new_idx_ls}')
            cnt+=1
            i+=1
        else:
            i+=1
    if verbose:  
        print(f'The combined bounds are \n {new_idx_ls}.')
    
    return new_idx_ls

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024]
                                      )
    
    o3d.visualization.draw_geometries([inlier_cloud])
    return True

#Observe the profile of a horizontal big-slice from front view.
def findVerticalBndPerRowSeg(obs_pts, slice_width, gap_th, ifPlot=True):
    '''Observe the profile of a horizontal big-slice from front view.'''

    v_sliced_pts = slicePointCloud(slice_width, las_pts=obs_pts, slice_axis=0) 
    sliceRprs = []
    
    for slice_pts in v_sliced_pts: 
        if len(slice_pts)>10:
            sliceRpr = 1
        else:
            #fill the near-empty cells with zeros.
            sliceRpr = 0
        sliceRprs.append(sliceRpr)
    
    #Next, find the longest consecutive ones and return the bounds' indices.
    rprs_col_min_idx = findConsecutiveOnes(sliceRprs, 
                                           thresh_num=gap_th//slice_width, 
                                           verbose=False)
    # print(f'rprs_col_min_idx={rprs_col_min_idx}')
    
    if ifPlot:
        plotMapWithMaxMinHorizontal(sliceRprs, rprs_col_min_idx, label='sum of Z')
    
    return rprs_col_min_idx
#%%
"""===========================MAIN PROGRAM BELOW============================"""
pts_time_stamps = ['08051158', '08121245', '08141222', '08161204', '08201154']
pts_tstp = pts_time_stamps[4]
pts_dir = rf'F:\2019snapbeans\lidar\2019reprocessed\2019{pts_tstp[:4]}\{pts_tstp[4:]}\forYieldPaper\{pts_tstp}_res_i_for_seg_clean.las'
# #%%
# #laspy 1.0
# from laspy.file import File
# f = File(pts_dir, mode='r')
# las_x = f.x - f.x.min()
# las_y = f.y - f.y.min()
# las_z = f.z - f.z.min()

# las_xyz = np.vstack((las_x, las_y, las_z)).T
# las_pts = np.hstack((las_xyz, f.i[:, np.newaxis])) #x, y, z, intensity


#laspy 2.0
f = laspy.read(pts_dir)
print(len(f.points))
las_x = f.x - f.x.min()
las_y = f.y - f.y.min()
las_z = f.z - f.z.min()
las_xyz = f.xyz - np.min(f.xyz, axis=0)
las_pts = np.hstack((las_xyz, f.i[:, np.newaxis])) #x, y, z, intensity
    
    
# =============================================================================
# #%%
# # =============================================================================
# # Calculate the boundaries and save them.
# # =============================================================================
# #Separate the points into columns
# x_split_coords = [0, 17, 34, 50, 67, 83, 100.6] #observed in point cloud, find the centers of the gaps among the rows.
# slice_wd = np.arange(0.02, 0.12, 0.02)
# slice_width = slice_wd[0]
# las_pts_abvgrd = las_pts[las_z>0.25] #The threshold is larger than the 2020 data because the residual values of the 2019 data could be as low as -0.08m.
# col_masks = [np.logical_and(las_pts_abvgrd[:, 0] > x_split_coords[i], 
#                             las_pts_abvgrd[:, 0] < x_split_coords[i+1]) for i in range(6)]
# 
# 
# las_pts_abvgrd_cols = [las_pts_abvgrd[col_mask] for col_mask in col_masks] 
# y_limits_ls, x_limits_ls = [], []
# for col_idx, obs_col in enumerate(las_pts_abvgrd_cols):
#     obs_col = las_pts_abvgrd_cols[col_idx]
#     pts_num_th = [80, 100, 200, 100, 50, 50]
#     thresh_num = [0, 0, 0, 0, 0, 0]
#     #%%
#     display3d = False
#     if display3d:
#         displayPointCloud(obs_col[:, :3])
#     
#     #%%
#     print("Statistical oulier removal")
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(obs_col[:, :3])   
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
#                                              std_ratio=2.0)
#     if display3d:
#         display_inlier_outlier(pcd, ind)
#     
#     obs_col_clean = obs_col[ind]
#     sliced_pts = slicePointCloud(slice_width, obs_col_clean, slice_axis=1) #cut the points into horizontal slices
#     
#     
#     #%%
#     #extract the up and bottom boundaries of each row.
#     rprs_col_map = []
#     
#     for slice_pts in sliced_pts: 
#         if len(slice_pts)>pts_num_th[col_idx]:
#                 sliceRpr = 1
#         else:
#             #fill the near-empty cells with zeros.
#             sliceRpr = 0
#         rprs_col_map.append(sliceRpr)
#         
#     rprs_col_min_idx = findRowOnes(rprs_col_map, thresh_num[col_idx])
#     las_y_abvgrd = obs_col_clean[:, 1]
#     y_limits = [[las_y_abvgrd.min()+idx*slice_width for idx in idxs] for idxs in rprs_col_min_idx]
#     if col_idx==2:
#         y_limits.pop(10)
#     
#     
#     plt.figure()
#     plt.scatter(obs_col_clean[:, 0], obs_col_clean[:, 1])
#     plt.hlines(y=np.asarray(y_limits, dtype=float).ravel(), xmin=min(obs_col_clean[:, 0]), xmax=max(obs_col_clean[:, 0]), color='r')
#     plt.show()
#     
#     #%%
#     #find all the left and right boundaries of each row.
#     h_slice_w = 0.02
#     v_slice_w = 0.2
#     gap_th = 1.5 #with unit meter.
#     las_x_abvgrd = obs_col_clean[:, 0]
#     v_rprs_col_bounds = []
#     for i, y_lim in enumerate(y_limits):
#         if y_lim[0] == y_lim[1]:
#             v_rprs_col_bounds.append(v_rprs_col_bounds[-1])
#             continue
#         down_limit = y_lim[0]
#         up_limit = y_lim[1]
#         obs_r_pts_mask = np.logical_and(las_y_abvgrd<up_limit, las_y_abvgrd>down_limit)
#         obs_r_pts = obs_col_clean[obs_r_pts_mask]
#     
#         ifPlot=False
#         v_rprs_row_min_idx = findVerticalBndPerRowSeg(obs_r_pts, 
#                                                       v_slice_w, 
#                                                       gap_th,
#                                                       ifPlot)
#         x_limits = [obs_r_pts[:, 0].min()+idx*v_slice_w for idx in v_rprs_row_min_idx]
#         v_rprs_col_bounds.append(x_limits)
#         
# 
#     y_limits_ls.append(y_limits)
#     x_limits_ls.append(v_rprs_col_bounds)
# 
# #%%
# #save the boundaries of each row segment.

# sav_y_limits = np.asarray(y_limits_ls, dtype=float) + f.y.min()
# sav_x_limits = np.asarray(x_limits_ls, dtype=float) + f.x.min()
# bound_file = pts_dir[:-4]+'_bound.pkl'
# write_dic = {"Data_format": "num_of_col*num_of_row*bounds",
#              "y_limits_ls": sav_y_limits,
#              "x_limits_ls": sav_x_limits}
# 
# with open(bound_file, 'wb') as bf:
#     pickle.dump(write_dic, bf)
#     
# print("The boudaries have been saved to a pkl file.")
# =============================================================================

#%%
#Load the boudaries from the file.
bound_file = r'F:\2019snapbeans\lidar\2019reprocessed\20190805\1158\forYieldPaper\08051158_res_i_for_seg_clean_bound.pkl'
with open(bound_file, 'rb') as bfr:
    pic = pickle.loads(bfr.read())

print(pic)


#%%
#loop over all the segments and assign labels to points in each segment.
seg_labels = np.zeros(len(las_pts), dtype=np.uint8)
y_limits_ls = pic['y_limits_ls'] - f.y.min()
x_limits_ls = pic['x_limits_ls'] - f.x.min()
i=1
for col_x_lim, col_y_lim in zip(x_limits_ls, y_limits_ls):
    for row_x_lim, row_y_lim in zip(col_x_lim, col_y_lim):
        down_limit = row_y_lim[0]
        up_limit = row_y_lim[1]
        left_limit = row_x_lim[0]
        right_limit = row_x_lim[1]
        seg_mask = np.logical_and.reduce((las_y<up_limit, 
                                        las_y>down_limit, 
                                        las_x>left_limit, 
                                        las_x<right_limit))
        print(f'Row {i}.')
        seg_labels[seg_mask] = i
        i+=1
        
plt.figure()
plt.hist(seg_labels[seg_labels>0], bins=96)

#%%
#save the segmentation labels. (laspy 1.0)
# from laspy.file import File
# h = f.header
# f2 = File(pts_dir[:-4]+'_seg.las', mode = "w", header=h)
# f2.points =  f.points
# f2.user_data = seg_labels
# f.close()
# f2.close()

#%%
#save the segmentation labels. (laspy 2.0)
f2 = laspy.LasData(f.header)
f2.points = f.points
f2.user_data = seg_labels
f2.write(pts_dir[:-4]+'_seg.las')

