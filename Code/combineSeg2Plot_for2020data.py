# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:
    This code combines the results of the representatives of row-segments 
    (72 samples) to plot-level (18 samples) by taking averages
    of the segment-representatives, and then save them to a txt file and 
    a excel file.

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
import time
import pandas as pd
start_time = time.time()


"""===========================MAIN PROGRAM BELOW============================"""
#%%
# LiDAR file name set-up
# flight_stamp = '08061055'
# file_path = fr'E:\2020snapbeans\2020{flight_stamp[:4]}\lidar\{flight_stamp[4:]}\rowSegmentation\{flight_stamp}_lidar_N_for_row_seg_seg_rprs.csv'

flight_stamps = ['07281204', '07311057', '08061055', 
                '08101049', '08141148', '08211044', '08241132']
# fl_stp = flight_stamps[0]
for fl_stp in flight_stamps:
    file_path = fr'F:\2020snapbeans\2020{fl_stp[:4]}\lidar\{fl_stp[4:]}\rowSegmentation\{fl_stp}_lidar_N_for_row_seg_seg_rprs.csv'
    print(file_path)
    # #%%
    # # MSI file name set-up
    # method = 'med'
    # # file_path = fr'E:\2020snapbeans\20200728\micasense\1204\pix4d\202007281204raw\4_index\reflectance\enviOutput\07281204_bgrren_VIs_mask_rotate_linear_row_{method}.csv'
    # file_path = fr'C:\ResearchData\2020snapbeans\20200810\micasense\1049\202008101049\4_index\reflectance\enviOutput\08101049_bgrren_VIs_mask_rotate_linear_row_{method}.csv'
    
    
    
    
    #%%
    in_df = pd.read_csv(file_path, index_col=0)
    cultivar_names = ['Venture', 'Huntington', 'Colter', 'Cabot', 'FlavorSweet', 'Denver']
    cultivar_chr = ['L1', 'L2', 'F1', 'F2', 'W1', 'W2']
    rep_names = [1, 2, 3]
    cultivar_dict = {'rep1':{'L1': [14, 17, 20, 23],
                             'L2': [13, 16, 19, 22],
                             'F1': [12, 15, 18, 21],
                             'F2': [26, 29, 32, 35],
                             'W1': [25, 28, 31, 34],
                             'W2': [24, 27, 30, 33]},
                     'rep2':{'L1': [49, 52, 55, 58],
                             'L2': [50, 53, 56, 59],
                             'F1': [48, 51, 54, 57],
                             'F2': [37, 40, 43, 46],
                             'W1': [36, 39, 42, 45],
                             'W2': [38, 41, 44, 47]},
                     'rep3':{'L1': [73, 76, 79, 82],
                             'L2': [60, 63, 66, 69],
                             'F1': [62, 65, 68, 71],
                             'F2': [74, 77, 80, 83],
                             'W1': [72, 75, 78, 81],
                             'W2': [61, 64, 67, 70]}
                     }
    print(in_df.columns)
    ave_cultivar_combined_dict = {key: {c_key: [] for c_key in value.keys()} for key, value in cultivar_dict.items()}
    
    for col_name in in_df.columns:
        for rep_key, plot_dict in cultivar_dict.items():
            for cult_key, seg_nums in plot_dict.items():
                seg_ave = np.around(np.average(in_df.loc[seg_nums, col_name]), decimals=3)
                ave_cultivar_combined_dict[rep_key][cult_key].append(seg_ave)
                
    #%%
    #write the combined resutls to txt file.
    with open(f"{file_path[:-4]}_combine.txt", 'w') as f: 
        f.write(f'Representative names: {list(in_df.columns)}\n\n')
        f.write('Average of every 4 segments within on sub-plot.\n')
        for rep_key, plot_dict in ave_cultivar_combined_dict.items(): 
            value_wr = '\n'.join([f'{c_key}:{str(value)}' for c_key, value in plot_dict.items()])
            f.write('%s:\n%s\n' % (rep_key, value_wr))   
            
    #%%
    #write the combined results to excel file.
    import xlsxwriter #this package only supports writing data, not reading data
    
    workbook = xlsxwriter.Workbook(f'{file_path[:-4]}_combine.xlsx')
    def write2ExcelSheet(workbook, sheet_name, in_dict, repr_names):
        worksheet = workbook.add_worksheet(sheet_name)
        
        #write the titles of the replications. 
        merge_format = workbook.add_format({'bold':     True,
                                            'border':   6,
                                            'align':    'center',
                                            'valign':   'vcenter',
                                            })
        for i, rep_key in enumerate(in_dict.keys()):
            worksheet.merge_range(f'A{2+i*6}:A{1+(1+i)*6}', rep_key, merge_format)
        
        # write the titles of the representatives horizontally. 
        #for metrics generated from LiDAR
        for i, col_name in enumerate(repr_names):
            if ord('C')+i<=90:
                cell_loc = f"{chr(ord('C')+i)}1"
            else:
                cell_loc = f"A{chr(ord('C')+i-26)}1"
            worksheet.write(cell_loc, col_name)
        
        
        # #for the VIs generated from MSI
        # for i, col_name in enumerate(repr_names):
        #     cell_col_name = str(col_name).rpartition('(')[2][:-1] 
        #     if ord('C')+i<=90:
        #         cell_loc = f"{chr(ord('C')+i)}1"
        #     else:
        #         cell_loc = f"A{chr(ord('C')+i-26)}1"
        #     worksheet.write(cell_loc, cell_col_name)
        
        
        #write the titles of the plots vertically. 
        for i, plot_dict in enumerate(in_dict.values()):
            for j, plot_key in enumerate(plot_dict.keys()):
                worksheet.write(f"B{2+i*6+j}", plot_key)
        
        # write the data to the file.
        for k, plot_dict in enumerate(in_dict.values()):
            for j, rpr_ls in enumerate(plot_dict.values()):
                for i,ele in enumerate(rpr_ls):
                    if ord('C')+i<=90:
                        cell_loc = f"{chr(ord('C')+i)}"
                    else:
                        cell_loc = f"A{chr(ord('C')+i-26)}"
                    worksheet.write(f"{cell_loc}{j+2+k*len(plot_dict)}", ele)
        return True
    
    write2ExcelSheet(workbook, 'Ave', ave_cultivar_combined_dict, in_df.columns)
    workbook.close()
    #%%
    print("--- %.1f seconds ---" % (time.time() - start_time))