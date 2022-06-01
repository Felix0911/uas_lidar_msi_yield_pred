# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:
    1. 

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
start_time = time.time()


"""===========================MAIN PROGRAM BELOW============================"""

#%%
import os
pc1 = r'E:\2020snapbeans\20200728\lidar\1204\lidar_spat_cropped_rn1_remDup005_SOR_10_1_Tshft_N_trim_small_angle20.csv'
pc_dem = r'E:\2020snapbeans\20200701\lidar\1150\lidar_N_trim_small_angle20_dem.csv'
os.system("echo Hello from the other side!")
calc_c2c_distance_command = f'CloudCompare -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT LAS -o -SKIP 1 -GLOBAL_SHIFT -334248.00 -4748090.00 -178.69 {pc1} -o {pc_dem} -c2c_dist '
#%%
print("--- %.1f seconds ---" % (time.time() - start_time))