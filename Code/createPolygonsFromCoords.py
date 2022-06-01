# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:
    Read the coordinates of the corners of the polygon and then create polygons
    and save them.
    1) read the coordinates from the .xlsx file and save them in a dictionary;
    2) create polygons and save.

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
import pandas as pd
start_time = time.time()


"""===========================MAIN PROGRAM BELOW============================"""
polygon_corners_path = 'polgonCorners.xlsx'
corners_df = pd.read_excel(polygon_corners_path)
corners_dic = {'P1':np.asarray(corners_df.loc[0:4, 'x':'y']),
               'P2':np.asarray(corners_df.loc[5:9, 'x':'y']),
               'P3':np.asarray(corners_df.loc[10:14, 'x':'y']),
               }


#%%
from osgeo import ogr
multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

item1 = corners_dic['P1']
rings = [ogr.Geometry(ogr.wkbLinearRing) for i in range(3)]
polys = [ogr.Geometry(ogr.wkbPolygon) for i in range(3)]
for i in range(3):
    for xy in corners_dic[f'P{i+1}']:
        rings[i].AddPoint(xy[0], xy[1])


    polys[i].AddGeometry(rings[i])
    multipolygon.AddGeometry(polys[i])

wkt = multipolygon.ExportToWkt()
print(wkt)
#%%
from osgeo import ogr
from osgeo import osr
# Set up the shapefile driver 
driver = ogr.GetDriverByName("ESRI Shapefile")

# create the data source
ds = driver.CreateDataSource("polygons.shp")

# create the spatial reference system, WGS84
srs =  osr.SpatialReference()
# srs.ImportFromEPSG(4326)

# create one layer 
layer = ds.CreateLayer("polygon", srs, ogr.wkbLineString)

# Add an ID field
idField = ogr.FieldDefn("id", ogr.OFTInteger)
layer.CreateField(idField)

# Create the feature and set values
featureDefn = layer.GetLayerDefn()
feature = ogr.Feature(featureDefn)
feature.SetGeometry(multipolygon)
feature.SetField("id", 1)
layer.CreateFeature(feature)

feature = None

# Save and close DataSource
ds = None

#%%
print("--- %.1f seconds ---" % (time.time() - start_time))