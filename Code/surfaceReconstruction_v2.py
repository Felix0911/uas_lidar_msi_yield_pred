# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:

Version: 1.0

Reference:
"""




import numpy as np
import matplotlib.pyplot as plt
import time
import open3d as o3d

from scipy.spatial import ConvexHull
import random
from shapely.geometry import Polygon, Point


def random_points_within(hull_vertices, num_points=100):
    '''Generate points randomly within a polygon.
    Input: - '''
    random.seed(2172020)
    poly = Polygon(hull_vertices)
    min_x, min_y, max_x, max_y = poly.bounds

    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            # points.append(random_point)
            points.append(np.asarray(random_point.xy).T)

    return np.vstack(points)

def createLeftFace(las_xyz, ifPlotHull=True):
    '''
    step 1: extract a slice of points and then project it to the plane x=0;
    step 2: find the convex hull of the projected points in step 1;
    step 3: make a planar face from the convex hull in step 2.
    '''
    inquiry_slice_width = 0.05
    slice_mask = las_xyz[:, 0] < inquiry_slice_width
    project_pts = las_xyz[slice_mask][:,1:]
    hull = ConvexHull(project_pts)
    if ifPlotHull:
        plt.figure()
        plt.plot(project_pts[:,0], project_pts[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(project_pts[simplex, 0], project_pts[simplex, 1], 'k-')    
        plt.plot(project_pts[hull.vertices,0], project_pts[hull.vertices,1], 'r--', lw=2)
        plt.plot(project_pts[hull.vertices[0],0], project_pts[hull.vertices[0],1], 'ro')
        plt.show() 
    
    plane_pts = random_points_within(project_pts[hull.vertices],100)
    left_face_pts = np.hstack((np.ones((len(plane_pts), 1)), plane_pts))
    return left_face_pts

def createRightFace(las_xyz, ifPlotHull=True):
    '''
    step 1: extract a slice of points and then project it to the plane x=x_max;
    step 2: find the convex hull of the projected points in step 1;
    step 3: make a planar face from the convex hull in step 2.
    '''
    inquiry_slice_width = 0.05
    slice_mask = las_xyz[:, 0]>(las_xyz[:, 0].max() - inquiry_slice_width)
    project_pts = las_xyz[slice_mask][:,1:]
    hull = ConvexHull(project_pts)
    if ifPlotHull:
        plt.figure()
        plt.plot(project_pts[:,0], project_pts[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(project_pts[simplex, 0], project_pts[simplex, 1], 'k-')    
        plt.plot(project_pts[hull.vertices,0], project_pts[hull.vertices,1], 'r--', lw=2)
        plt.plot(project_pts[hull.vertices[0],0], project_pts[hull.vertices[0],1], 'ro')
        plt.show() 
    
    plane_pts = random_points_within(project_pts[hull.vertices],100)
    right_face_pts = np.hstack((np.ones((len(plane_pts), 1))*las_xyz[:, 0].max(), plane_pts))
    return right_face_pts

def createBottomFace(las_xyz):
    bottom_face_pts = las_xyz.copy()
    p = 2
    bottom_face_pts[:, 2] = np.ones(len(las_xyz))*np.percentile(bottom_face_pts[:, 2], p)
    return bottom_face_pts

def cleanMeshO3D(o3d_mesh, ifDownSamp=False, downSampNum=20_000):
    if ifDownSamp: #decimating the mesh to certain number of triangles
        out_mesh = o3d_mesh.simplify_quadric_decimation(downSampNum)
    else:
        out_mesh = o3d_mesh
        
    out_mesh.remove_degenerate_triangles()
    out_mesh.remove_duplicated_triangles()
    out_mesh.remove_duplicated_vertices()
    out_mesh.remove_non_manifold_edges()
    
    return out_mesh
  

if __name__ == '__main__':
    #%%
    start_time = time.time()
    #laspy 1.0
    from laspy.file import File
    """===========================MAIN PROGRAM BELOW============================"""
    pts_dir = r'E:\2020snapbeans\20200728\lidar\1143\plantCount\crop_07281143_rotated.las'
    f = File(pts_dir, mode='r')
    las_x = f.x - f.x.min()
    las_y = f.y - f.y.min()
    las_z = f.z - f.z.min()
    las_xyz = np.vstack((las_x, las_y, las_z)).T
    
    
    # #%%
    # #laspy 2.0
    # import laspy
    # """===========================MAIN PROGRAM BELOW============================"""
    # # pts_dir = r'E:\2020snapbeans\20200728\lidar\1143\plantCount\crop_07281143_rotated.las'
    # pts_dir = r'G:\My Drive\code\plantCount\car_clean.las'
    # f = laspy.read(pts_dir)
    # # las_xyz = f.xyz -   
    
    #%%
    #add surrounding faces to the point cloud
    
    left_face_pts = createLeftFace(las_xyz)
    right_face_pts = createRightFace(las_xyz)
    bottom_face_pts = createBottomFace(las_xyz)
    plt.figure()
    plt.scatter(right_face_pts[:,1], right_face_pts[:,2])
    
    
    #%%
    # def addFaces(las_xyz):
    #     '''Add left, right, and bottom faces to the point cloud segment.'''
    #     left_mask = las_xyz[:, 0]<0.1
    #     left_face_pts = las_xyz[left_mask]
    #     left_face_pts[:, 0] = np.zeros_like(left_face_pts[:, 0])
        
    #     right_mask = las_xyz[:, 0]>(las_xyz[:, 0].max()-0.1)
    #     right_face_pts = las_xyz[right_mask]
    #     right_face_pts[:, 0] = np.ones_like(right_face_pts[:, 0])*las_xyz[:, 0].max()
        
    #     bottom_face_pts = las_xyz.copy()
    #     bottom_face_pts[:, 2] = np.zeros_like(bottom_face_pts[:, 2])
        
    #     return left_face_pts, right_face_pts, bottom_face_pts
    
    # left_face_pts, right_face_pts, bottom_face_pts = addFaces(las_xyz)
    #%%
    extended_abv_grd_pts = np.vstack((las_xyz, 
                                  left_face_pts, 
                                  right_face_pts, 
                                  # bottom_face_pts
                                  ))
    # extended_abv_grd_pts = las_xyz
    #calculate normals
    pcd_offgrd = o3d.geometry.PointCloud()
    pcd_offgrd.points = o3d.utility.Vector3dVector(extended_abv_grd_pts)
    pcd_offgrd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30))
    pcd_offgrd.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])
    o3d.visualization.draw_geometries([pcd_offgrd])
    
    pcd_grd = o3d.geometry.PointCloud()
    pcd_grd.points = o3d.utility.Vector3dVector(bottom_face_pts)
    pcd_grd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30))
    pcd_grd.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, -1.0])
    o3d.visualization.draw_geometries([pcd_grd])
    
    
    #%%
    #save points
    save_path = r'E:\2020snapbeans\20200728\lidar\1143\plantCount\crop_07281143_rotated_extended.txt'
    extended_pts_normals = np.vstack((np.asarray(pcd_offgrd.normals), np.asarray(pcd_grd.normals)))
    extended_pts_xyz = np.vstack((extended_abv_grd_pts, bottom_face_pts))
    np.savetxt(save_path, np.hstack((extended_pts_xyz, extended_pts_normals)))
    
    #%%
    pcd_overall = o3d.geometry.PointCloud()
    pcd_overall.points = o3d.utility.Vector3dVector(extended_pts_xyz)
    pcd_overall.normals = o3d.utility.Vector3dVector(extended_pts_normals)
    o3d.visualization.draw_geometries([pcd_overall])
    #%%
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_overall)
    print(mesh)
    print("Painting the mesh")
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])
    # mesh.remove_non_manifold_edges()
    print(f'The surface area of the mesh is :{o3d.geometry.TriangleMesh.get_surface_area(mesh):.2f}')
    # mesh.remove_duplicated_triangles()
    # o3d.geometry.TriangleMesh.get_volume(mesh)
    #%%
    mesh.compute_triangle_normals()
    save_mesh_path = 'crop_07281143_rotated_mesh.stl'
    o3d.io.write_triangle_mesh(save_mesh_path, mesh)
    
    # #%%
    # print('filter with average with 1 iteration')
    # mesh_out = mesh.filter_smooth_simple(number_of_iterations=1)
    # mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out])
    # save_mesh_path = 'crop_07281143_rotated_mesh_itr1.stl'
    # o3d.io.write_triangle_mesh(save_mesh_path, mesh_out)
    # #%%
    # print('filter with average with 5 iterations')
    # mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
    # mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out])
    # save_mesh_path = 'crop_07281143_rotated_mesh_itr5.stl'
    # o3d.io.write_triangle_mesh(save_mesh_path, mesh_out)
    
    # #%%
    # print('filter with average with 1000 iterations')
    # mesh_out = mesh.filter_smooth_simple(number_of_iterations=1000)
    # mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out])
    # save_mesh_path = 'crop_07281143_rotated_mesh_itr1000.stl'
    # o3d.io.write_triangle_mesh(save_mesh_path, mesh_out)
    
    #%%
    #Convex Hull
    convex_pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
    hull, _ = convex_pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([convex_pcl, hull_ls])
    #%%
    save_mesh_path = 'crop_07281143_rotated_mesh_convex.stl'
    hull.compute_vertex_normals()
    o3d.io.write_triangle_mesh(save_mesh_path, hull)
    
    
    # #%%
    # #Ball-Pivoting Algorithm (BPA)
    # distances = pcd_overall.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 10 * avg_dist
    
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_overall,
    #                                                                            o3d.utility.DoubleVector([radius, radius * 2]))
    # o3d.visualization.draw_geometries([bpa_mesh])
    
    
    #%%
    #Alpha shape 
    # pcd_overall = o3d.geometry.PointCloud()
    pcd_alpha = mesh.sample_points_poisson_disk(15000)
    o3d.visualization.draw_geometries([pcd_alpha])
    #%%
    alpha = 0.1
    print(f"alpha={alpha:.3f}")
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_overall)
    # alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_overall, alpha, tetra_mesh, pt_map)
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_alpha, alpha)
    
    alpha_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)
    #%%
    print("--- %.1f seconds ---" % (time.time() - start_time))