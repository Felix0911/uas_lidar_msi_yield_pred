# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 00:16:48 2022

@author: Fei Zhang
"""

import trimesh
import numpy as np
import surfaceReconstruction_v2 as srv2
import open3d as o3d
from scipy.interpolate import griddata
import copy

class CalcRprstPerSlice:
    '''This class calculates different representatives of the slices of the
    snap bean rows.
    '''
    def __init__(self, slice_pts):
        self.slice_pts = slice_pts
        self.las_xyz = slice_pts[:,:3]
        self.row_length = slice_pts[:, 0].max() - slice_pts[:, 0].min()
        
    def calc90Percentile(self):
        '''Calculate 90th percentile of the z coordinates'''
        out_h = np.percentile(self.slice_pts[:, 2], 90)
        return out_h
    
    def calcSumZ(self):
        out_sum = np.sum(self.slice_pts[:,2]) / self.row_length
        return out_sum
    
    def calcNum(self):
        out_num = len(self.slice_pts) / self.row_length
        return out_num
    
    def calcSumIntns(self):
        out_sum = np.sum(self.slice_pts[:,3]) / self.row_length
        return out_sum
    
    def calcTopArea(self, h_th=95, grid_size=0.02):   
        '''Find the top points and then rasterize them and then calculate
        the top area. Finally, normalized by length'''
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
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        # grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0.0)
        top_area = len(grid_z0[grid_z0>h_p])*(grid_size**2)
        top_area_per_length = top_area / self.row_length
        return top_area_per_length 
    
    
    def calcPoissonRecVolume(self, temp_mesh_path, addLeftRight=False, depth=9, ifVisual=False):
        '''
        step 1: Extend the point clouds and add normals
        step 2: Poisson reconstruction using Open3D.
        step 3: Clean the mesh
        step 4: Calculate the reconstructed mesh volume using trimesh.
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

        
        #step 3
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        mesh_clean = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 300
        mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        if ifVisual:
            mesh_clean.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries([mesh_clean])
        
        o3d.io.write_triangle_mesh(temp_mesh_path, mesh_clean)
        
        
        #step 4
        surface_area = mesh_clean.get_surface_area() / self.row_length
        mesh_tm = trimesh.load(temp_mesh_path)
        volume = mesh_tm.volume / self.row_length
        #compare the volume of the mesh with the volume of its convex hull
        vol_ratio = mesh_tm.volume / mesh_tm.convex_hull.volume
        
        return surface_area, volume, vol_ratio
    
    def calcLidarLAI(self, h_th=0.25):
        '''Calculate the Laser Penetration Index.
        LPI = R_T / R_total, where R_T is the number of the points below a 
        height threshold.
        LAI = sqrt(LPI). '''
        # h_p = np.percentile(self.slice_pts[:, 2], h_th)
        bottom_mask = self.slice_pts[:, 2]<h_th
        bottom_pts = self.slice_pts[bottom_mask]
        R_T = len(bottom_pts)
        LPI = R_T / len(self.slice_pts)
        LAI = 4.73-4.51*np.sqrt(LPI) #Ref: Fei 2022 LAI paper.
        
        return LAI