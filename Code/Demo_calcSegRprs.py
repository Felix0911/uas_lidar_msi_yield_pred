# -*- coding: utf-8 -*-
"""
Created on June 1 2022

Description: This code read the point clouds with labels and then calculate 
representatives (top canopy area, poisson mesh volume) of a example segment. 

@author: Fei Zhang
"""


import trimesh
import numpy as np
import matplotlib.pyplot as plt
import surfaceReconstruction_v2 as srv2
import open3d as o3d
from scipy.interpolate import griddata
import pandas as pd
import laspy
#%%
 
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
        top_area = len(grid_z0[grid_z0>h_p])*(grid_size**2)
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





#%%
"""===========================MAIN PROGRAM BELOW============================"""
pts_dir = r'F:\2020snapbeans\20200824\lidar\1132\rowSegmentation\08241132_lidar_N_for_row_seg_seg.las'
#%%
#laspy 1.0
# from laspy.file import File
# f = File(pts_dir, mode='r')
# las_x = f.x - f.x.min()
# las_y = f.y - f.y.min()
# las_z = f.z - f.z.min()

# las_xyz = np.vstack((las_x, las_y, las_z)).T

f = laspy.read(pts_dir)
print(len(f.points))
las_x = f.x - f.x.min()
las_y = f.y - f.y.min()
las_z = f.z - f.z.min()
las_xyz = f.xyz - np.min(f.xyz, axis=0)
# las_pts = np.hstack((las_xyz, f.i[:, np.newaxis])) #x, y, z, intensity


seg_labels = f.user_data
las_pts = np.hstack((las_xyz, f.i[:, np.newaxis], seg_labels[:, np.newaxis])) #x, y, z, intensity, row_seg_label

for spec in f.point_format:
    print(spec.name)


#%%
labeled_seg_pts = []
for i in range(1, seg_labels.max()):
    seg_pts = las_pts[seg_labels==i]
    labeled_seg_pts.append(seg_pts)
    
#%%
# Observe the impact of the h_th in calculating the topArea
selected_seg_pts = labeled_seg_pts[:]
h_th_ls = np.arange(70, 98)
topArea_ls = [[] for s in selected_seg_pts]
grid= 0.03

for i, seg_pts in enumerate(selected_seg_pts):
    for j, h_th in enumerate(h_th_ls):
        cp_pts = CalcRprstPerSlice(seg_pts)
        topArea = cp_pts.calcTopArea(h_th, grid_size=grid)
        topArea_ls[i].append(topArea)
    print(f'segment {i} finished!')
  
#%%
# title = f'Change of topArea over h_th (grid={grid})'
plt.rcParams.update({'font.size': 15})
plt.figure()
for i in range(len(selected_seg_pts)):
    plt.plot(h_th_ls, topArea_ls[i], linestyle='-', marker='.')    
plt.xlabel('Height threshold (percentile)')
plt.ylabel('Top canopy area (${m^2}$)')
plt.grid()
# plt.legend()
# plt.title(title)
plt.tight_layout()
plt.show()

topArea_ls_arr = np.asarray(topArea_ls)
ta_std_per_h = np.std(topArea_ls_arr, axis=0)
plt.figure()
plt.plot(h_th_ls, ta_std_per_h, linestyle='-', marker='.')  
plt.xlabel('Height threshold (percentile)')
plt.ylabel('Std of the TCA (${m^2}$)')
plt.grid()
best_h_th = h_th_ls[np.argmax(ta_std_per_h)]
plt.axvline(x=95, color='r')
# plt.annotate(f'h_th={best_h_th}', (best_h_th, max(ta_std_per_h)))
# plt.legend()
# plt.title(title)
plt.tight_layout()
plt.show()

ta_mean_per_h = np.mean(topArea_ls_arr, axis=0)
plt.figure()
plt.plot(h_th_ls, ta_mean_per_h, linestyle='-', marker='.')    
plt.xlabel('Height threshold (percentile)')
plt.ylabel('Mean of the TCA (${m^2}$)')
plt.grid()
# plt.legend()
# plt.title(title)
plt.tight_layout()
plt.show()


# #%%       
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