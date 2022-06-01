# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:44:28 2022

Description: This code segments the north plot of 2022 snap beans data into 
row segments. It first does the row segments on the above ground points, which
were extracted by using a hight threshold on z-residual from the rotated and
markers-removed point cloud. Then, it calculated representatives for all the
segments (96 samples).

@author: Fei Zhang
"""


import trimesh
import numpy as np
import matplotlib.pyplot as plt
import surfaceReconstruction_v2 as srv2
import open3d as o3d
from scipy.interpolate import griddata
import pandas as pd
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
    
class CalcRprstPerSlice:
    '''This class calculates different representatives of the slices of the
    snap bean rows.
    '''
    def __init__(self, slice_pts):
        self.slice_pts = slice_pts
        self.las_xyz = slice_pts[:,:3]
        
    def calc90Percentile(self):
        '''Calculate 90th percentile of the z coordinates'''
        out_h = np.percentile(self.slice_pts[:, 2], 85)
        return out_h
    
    def calcSumZ(self):
        out_sum = np.sum(self.slice_pts[:,2])
        return out_sum
    
    def calcNum(self):
        out_num = len(self.slice_pts)
        return out_num
    
    def calcSumIntns(self):
        out_sum = np.sum(self.slice_pts[:,3])
        return out_sum
    
    def calcTopArea(self, h_th=85, grid_size=0.02):   
        '''Find the top points and then rasterize them and then calculate
        the top area.'''
        h_p = np.percentile(self.slice_pts[:, 2], h_th)
        top_mask = self.slice_pts[:, 2]>h_p
        top_pts = self.slice_pts[top_mask]

        x_min = np.min(top_pts[:,0])
        x_max = np.max(top_pts[:,0])
        y_min = np.min(top_pts[:,1])
        y_max = np.max(top_pts[:,1])
        grid_x, grid_y = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size]
        points = top_pts[:,:2]
        values = top_pts[:,2]
        # grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0.0)
        top_area = np.sum(grid_z0[grid_z0>h_p])*(grid_size**2)
        return top_area 
    
    
    def calcPoissonRecVolume(self, addLeftRight=False, depth=9):
        '''
        step 1: Extend the point clouds and add normals
        step 2: Poisson reconstruction using Open3D.
        step 3: Calculate the reconstructed mesh volume using trimesh.
        Tuning parameters: the possion reconstruction params.
        '''
        #step 1
        bottom_face_pts = srv2.createBottomFace(self.las_xyz)
        if addLeftRight:
            left_face_pts = srv2.createLeftFace(self.las_xyz)
            right_face_pts = srv2.createRightFace(self.las_xyz)
            
            extended_abv_grd_pts = np.vstack((self.las_xyz, 
                                          left_face_pts, 
                                          right_face_pts
                                          ))
        else:
            extended_abv_grd_pts = self.las_xyz
        
        pcd_offgrd = o3d.geometry.PointCloud()
        pcd_offgrd.points = o3d.utility.Vector3dVector(extended_abv_grd_pts)
        pcd_offgrd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.1, max_nn=30))
        pcd_offgrd.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
        
        pcd_grd = o3d.geometry.PointCloud()
        pcd_grd.points = o3d.utility.Vector3dVector(bottom_face_pts)
        pcd_grd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.1, max_nn=30))
        pcd_grd.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, -1.0])
        
        extended_pts_normals = np.vstack((np.asarray(pcd_offgrd.normals), 
                                          np.asarray(pcd_grd.normals)))
        extended_pts_xyz = np.vstack((extended_abv_grd_pts, bottom_face_pts))
        
        pcd_overall = o3d.geometry.PointCloud()
        pcd_overall.points = o3d.utility.Vector3dVector(extended_pts_xyz)
        pcd_overall.normals = o3d.utility.Vector3dVector(extended_pts_normals)
        
        #step 2
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_overall, depth=depth)
        mesh.compute_triangle_normals()
        save_mesh_path = r'G:\My Drive\code\plantCount\temp.stl'
        o3d.io.write_triangle_mesh(save_mesh_path, mesh)
        
        #step 3
        mesh_tm = trimesh.load(save_mesh_path)
        volume = mesh_tm.volume
        #compare the volume of the mesh with the volume of its convex hull
        vol_ratio = mesh_tm.volume / mesh_tm.convex_hull.volume
        
        return mesh_tm, volume, vol_ratio


def plotMap(py, label):
    title = 'Finding row centers'
    plt.figure()
    px = np.linspace(0, las_y.max(), len(py))
    plt.plot(px, py, '.-', mfc='none', label=label)
    plt.title(title)
    plt.xlabel('y coordinate')
    plt.ylabel(label)
    plt.show()
    

def plotMapWithMaxMin(py, m_idx, label, title):
    '''Finding Maxima or Minima'''
    plt.figure()
    px = np.linspace(0, las_y.max(), len(py))
    plt.plot(px, py, '.-', mfc='none', label=label)
    plt.title(title)
    plt.xlabel('y coordinate')
    plt.ylabel(label)
    for m_i in m_idx:
        plt.axvline(x=px[m_i], c='r', linestyle='--')
    
    plt.legend()
    plt.show()
    


def findLocalMaximumHorizontal(in_vec, search_r=5):
    '''Find the local maxima of the representative maps, which are the centers
    of the rows.'''
    out_idx = []
    for i, ele in enumerate(in_vec):
        if i<search_r or i>(len(in_vec)-search_r):
            continue
        else:
            if ele>max(in_vec[i-search_r:i]) and ele>max(in_vec[i+1:search_r+i+1]):
                out_idx.append(i)
    print(out_idx)            
    return np.asarray(out_idx)



def findLocalMinimumHorizontal(in_vec, search_r=5):
    '''Find the local minima of the representative maps, which are the 
    boundaries of the rows, which fall in the gaps between rows.'''
    out_idx = []
    # previous = -1
    for i, ele in enumerate(in_vec):
        if i==0:
            #if it's the beginning, record it.
            out_idx.append(i)
        elif i<search_r or i>(len(in_vec)-search_r):
            # previous = ele
            continue
        else:
            if ele<min(in_vec[i-search_r:i]) and ele<min(in_vec[i+1:search_r+i+1]):
                #if it's a valley point, record it twice, as it's the boundary
                #of the current row and the next row. This also includes single
                out_idx.append(i)
                out_idx.append(i+1)
            elif ele==0 and in_vec[i-1]>0 and in_vec[i-2]>0: 
                #if it is the first zero within a valley, record it.
                out_idx.append(i)
                if in_vec[i+1]>0 and in_vec[i+2]==0:
                   out_idx.append(i+2)
            elif ele>0 and in_vec[i-1]==0 and in_vec[i-2]==0:
                #if it is the last zero within a valley, record it.
                out_idx.append(i-1)
            else:
                continue
                
    print(len(out_idx), out_idx)            
    return np.asarray(out_idx)


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def findContinuousZero(in_vec):
    zero_range = zero_runs(in_vec)
    zero_range_diff = zero_range[:,1] - zero_range[:,0]
    #find the ranges of two largest consecutive zeros
    gap_indices = np.sort(zero_range[np.argsort(zero_range_diff)[-2:]].ravel())
    print(f'gap_indices={gap_indices}')
    return list(gap_indices)

#Observe the profile of a horizontal big-slice from front view.
def findVerticalBoundary(obs_pts, slice_width=0.2, ifPlot=True):
    '''Observe the profile of a horizontal big-slice from front view.'''

    v_sliced_pts = slicePointCloud(slice_width, las_pts=obs_pts, slice_axis=0) 
    v_sumZ_map = []
    
    for slice_pts in v_sliced_pts: 
        if slice_pts.shape[0]>10:
            cp_pts = CalcRprstPerSlice(slice_pts)
            sumZ = cp_pts.calcSumZ()
        else:
            #fill the near-empty cells with zeros.
            sumZ = 0
        v_sumZ_map.append(sumZ)
    
    v_sumZ_min_idx = [0] + findContinuousZero(v_sumZ_map) + [len(v_sumZ_map)-1]
    print(f'v_sumZ_min_idx={v_sumZ_min_idx}')
    
    if ifPlot:
        plotMapWithMaxMin(v_sumZ_map, v_sumZ_min_idx, label='sumZ_map', title='Minima')
    
    return v_sumZ_min_idx


#%%
"""===========================MAIN PROGRAM BELOW============================"""
# pts_dir = r'E:\2020snapbeans\20200728\lidar\1204\rowSegmentation\07281204_lidar_N_for_row_seg.las'
pts_dir = r'C:\ResearchData\2020snapbeans\20200824\lidar\1132\rowSegmentation\08241132_lidar_N_for_row_seg.las'
#%%
#laspy 1.0
from laspy.file import File
f = File(pts_dir, mode='r')
las_x = f.x - f.x.min()
las_y = f.y - f.y.min()
las_z = f.z - f.z.min()

las_xyz = np.vstack((las_x, las_y, las_z)).T
las_pts = np.hstack((las_xyz, f.i[:, np.newaxis])) #x, y, z, intensity

for spec in f.point_format:
    print(spec.name)
#%%
# #laspy 2.0
# import laspy
# f = laspy.read(pts_dir)
# print(len(f.points))
# las_xyz = f.xyz - np.min(f.xyz, axis=0)

#%%
#Read the boundaries from the txt file.
import pickle
# bound_file = pts_dir[:-4]+'_bound.pkl'
bound_file = 'lidar_N_for_row_seg_bound.pkl'
with open(bound_file, 'rb') as bfr:
    data = bfr.read()
    
print("Data type before reconstruction : ", type(data))
      
pic = pickle.loads(data)
print("Data type after reconstruction : ", type(pic))
print(dict(pic))

y_limits = pic['y_limits'] - f.y.min()
v_sumZ_bounds = pic['x_limits'] - f.x.min()
#%%
#loop over all the segments and assign labels to points in each segment.
seg_labels = np.zeros(len(las_pts), dtype=np.uint8)
h_slice_w = 0.02
v_slice_w = 0.15
for i, y_lim in enumerate(y_limits):
    if i%2==0:
        down_limit = y_lim
        up_limit = y_limits[i+1]
        obs_r_pts_mask = np.logical_and(las_y<up_limit, las_y>down_limit)
    print(f'Row {i}.')
    for j,v_bound in enumerate(v_sumZ_bounds[i//2]):
        if j%2==0:
            left_limit = v_bound
            right_limit = v_sumZ_bounds[i//2][j+1]
            obs_c_pts_mask = np.logical_and(las_x<right_limit, las_x>left_limit)
            seg_mask = np.logical_and(obs_r_pts_mask, obs_c_pts_mask)
            print(f'Number of seg points: {np.sum(seg_mask)}')
            seg_labels[seg_mask] = np.intc(i//2*3 + j//2)
#%%    
plt.figure()
plt.hist(seg_labels, bins=96)


#%%
#save the segmentation labels.
h = f.header
f2 = File(pts_dir[:-4]+'_seg.las', mode = "w", header=h)
f2.points =  f.points
f2.user_data = seg_labels
f.close()
f2.close()


#%%
#calculate the representatives within each segment and then save the results.
results = []
for i in range(1, seg_labels.max()):
    seg_pts = las_pts[seg_labels==i]
    cp_pts = CalcRprstPerSlice(seg_pts)
    h_90pct = cp_pts.calc90Percentile()
    sumZ = cp_pts.calcSumZ()
    num = cp_pts.calcNum()
    cumInt = cp_pts.calcSumIntns()
    topArea = cp_pts.calcTopArea()
    _, poisson_vol, vol_ratio = cp_pts.calcPoissonRecVolume()
    print(i)
    results.append([i, h_90pct, sumZ, num, cumInt, topArea, poisson_vol, vol_ratio])
    
sav_df = pd.DataFrame(np.array(results), 
                      columns=['seg_index', 'h_90pct', 'sumZ', 'num', 'cumInt', 
                               'topArea', 'poisson_vol', 'vol_ratio'])   
sav_df.to_csv(pts_dir[:-4]+'_seg_rprs.csv', index=False) 
# #%%
# labeled_seg_pts = []
# for i in range(1, seg_labels.max()):
#     seg_pts = las_pts[seg_labels==i]
#     labeled_seg_pts.append(seg_pts)
    
# #%%
# # Observe the impact of the h_th in calculating the topArea
# h_th_ls = np.arange(70, 98)
# topArea_ls = [[] for h in h_th_ls]
# grid= 0.03
# for j, h_th in enumerate(h_th_ls):
#     for seg_pts in labeled_seg_pts:
#         cp_pts = CalcRprstPerSlice(seg_pts)
#         topArea = cp_pts.calcTopArea(h_th, grid_size=grid)
#         topArea_ls[j].append(topArea)
#     print(f'h_th={h_th} finished!')
      
# title = f'Change of topArea over h_th (grid={grid})'
# plt.figure()
# for h_th, topA in zip(h_th_ls, topArea_ls):
#     label = f'h_th={h_th}'
#     px = np.arange(0, len(topA))
#     plt.plot(px, topA, '.-', mfc='none', label=label)
    
# plt.xlabel('Segment index')
# plt.ylabel('Top Area')
# plt.title(title)
# plt.legend()
# plt.show()
        
# plt.figure()
# topArea_total_ls = [sum(topA) for topA in topArea_ls]
# label = 'Sum of topArea by h_th'
# plt.plot(h_th_ls, topArea_total_ls, '.-', mfc='none', label=label)
# plt.xlabel('h_th')
# plt.ylabel('Sum of topArea')
# plt.title(title)
# plt.legend()
# plt.grid()
# plt.show()

# #%%
# # Observe the impact of the grid_size in calculating the topArea
# grid_size_ls = np.arange(0.01, 0.11, 0.01)
# topArea_ls = [[] for grid in grid_size_ls]
# h_th = 85
# for j, grid in enumerate(grid_size_ls):
#     for seg_pts in labeled_seg_pts:
#         cp_pts = CalcRprstPerSlice(seg_pts)
#         topArea = cp_pts.calcTopArea(h_th=h_th, grid_size=grid)
#         topArea_ls[j].append(topArea)
#     print(f'Grid size={grid} finished!')
      

# title = 'Change of topArea over grid_size'
# plt.figure()
# for grid, topA in zip(grid_size_ls, topArea_ls):
#     label = f'grid_size={grid}'
#     px = np.arange(0, len(topA))
#     plt.plot(px, topA, '.-', mfc='none', label=label)
    
# plt.xlabel('Segment index')
# plt.ylabel('Top Area')
# plt.title(title)
# plt.legend()
# plt.show()
        
# plt.figure()
# topArea_total_ls = [sum(topA) for topA in topArea_ls]
# label = 'Sum of topArea by grid_size'
# plt.plot(grid_size_ls, topArea_total_ls, '.-', mfc='none', label=label)
    
# plt.xlabel('grid_size')
# plt.ylabel('Sum of topArea')
# plt.title(title)
# plt.legend()
# plt.grid()
# plt.show()


# #%%
# # Observe the impact of the h_th and grid_size in calculating the topArea
# topArea_sum_ls = [[] for grid in grid_size_ls]
# for k, grid in enumerate(grid_size_ls):
#     for j, h_th in enumerate(h_th_ls):
#         temp_ls = []
#         for seg_pts in labeled_seg_pts:
#             cp_pts = CalcRprstPerSlice(seg_pts)
#             topArea = cp_pts.calcTopArea(h_th=h_th, grid_size=grid)
#             temp_ls.append(topArea)
#         topArea_sum_ls[k].append(sum(temp_ls))
#         print(f'Grid size={grid:.2f} h_th={h_th} finished!')
# #%%
# title = 'Change of topArea over grid_size and h_th'
# plt.figure()
# for grid, topArea_total_ls in zip(grid_size_ls, topArea_sum_ls):
#     label = f'grid_size={grid:.2f}'
#     plt.plot(h_th_ls, topArea_total_ls, '.-', mfc='none', label=label)

# plt.xlabel('h_th')
# plt.ylabel('Sum of topArea')
# plt.title(title)
# plt.legend()
# plt.grid()
# plt.show()

# #%%
# # Observe the impact of the depth in calculating the Poisson volume
# poisson_vol_depth_ls = np.arange(8, 14, 1)
# ps_v_ls = [[] for dp in poisson_vol_depth_ls]
# v_ratio_ls = [[] for dp in poisson_vol_depth_ls]

# for j, dp in enumerate(poisson_vol_depth_ls):
#     for seg_pts in labeled_seg_pts:
#         cp_pts = CalcRprstPerSlice(seg_pts)
#         _, poisson_vol, vol_ratio = cp_pts.calcPoissonRecVolume(depth=dp)
#         ps_v_ls[j].append(poisson_vol)
#         v_ratio_ls[j].append(vol_ratio)
#     print(f'Depth = {dp} finished!')
      
# #%%
# title = 'Change of Poisson Volume over depth'
# plt.figure()
# for dp, ps_v in zip(poisson_vol_depth_ls, ps_v_ls):
#     v_std = np.std(ps_v)
#     v_mean = np.mean(ps_v)
#     label = f'Depth={dp}, std={v_std:.3f}, mean={v_mean:.3f}'
#     px = np.arange(0, len(ps_v))
#     plt.plot(px, ps_v, '.-', mfc='none', label=label)
    
# plt.xlabel('Segment index')
# plt.ylabel('Poisson Volume')
# plt.title(title)
# plt.legend()
# plt.show()

      
# plt.figure()
# ps_vol_total_ls = [sum(ps_v) for ps_v in ps_v_ls]
# label = 'Sum of Poisson Volume by depth'
# plt.plot(poisson_vol_depth_ls, ps_vol_total_ls, '.-', mfc='none', label=label)
    
# plt.xlabel('Poisson depth')
# plt.ylabel('Sum of Poisson Volume')
# plt.title(title)
# plt.ylim(ymin=0, ymax=300)
# plt.legend()
# plt.grid()
# plt.show()