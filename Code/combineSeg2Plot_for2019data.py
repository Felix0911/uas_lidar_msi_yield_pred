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
time_stamps = ['08051158', '08121245', '08141222', '08161204', '08201154']
# tstp = time_stamps[4]
for tstp in time_stamps:
    file_path = rf'F:\2019snapbeans\lidar\2019reprocessed\2019{tstp[:4]}\{tstp[4:]}\forYieldPaper\{tstp}_res_i_for_seg_clean_seg_rprs.csv'
    # file_path = rf'F:\2019snapbeans\SfM\snb_{tstp}\snb_{tstp}\4_index\reflectance\enviOutput\{tstp}_bgrren_VIs_mask_rotate_nn_row_ave.csv'
    
    
    #%%
    in_df = pd.read_csv(file_path, index_col=0)
    cultivar_names = ['Venture', 'Huntington', 'Colter', 'Cabot', 'FlavorSweet', 'Denver']
    cultivar_chr = ['L1', 'L2', 'F1', 'F2', 'W1', 'W2']
    # The following 3 dicts were prepared for making the cultivar_dict.
    # plot_lab_dic_6 = {'L1':'FOQX', 
    #                   'L2':'GNTU', 
    #                   'F1':'CEPV', 
    #                   'F2':'AJLW', 
    #                   'W1':'BHMS', 
    #                   'W2':'DIKR'}
    # rep_cult_dic = {'rep1':'DHLPTX',
    #                 'rep2':'CGKOSW',
    #                 'rep3':'BFJNRV',
    #                 'rep4':'AEIMQU',
    #                 }
    # cult_row_dic = {'A': [81, 82, 83, 84], 
    #                 'B': [85, 86, 87, 88], 
    #                 'C': [89, 90, 91, 92], 
    #                 'D': [93, 94, 95, 96], 
    #                 'E': [65, 66, 67, 68], 
    #                 'F': [69, 70, 71, 72], 
    #                 'G': [73, 74, 75, 76], 
    #                 'H': [77, 78, 79, 80], 
    #                 'I': [49, 50, 51, 52], 
    #                 'J': [53, 54, 55, 56], 
    #                 'K': [57, 58, 59, 60], 
    #                 'L': [61, 62, 63, 64], 
    #                 'M': [33, 34, 35, 36], 
    #                 'N': [37, 38, 39, 40], 
    #                 'O': [41, 42, 43, 44], 
    #                 'P': [45, 46, 47, 48], 
    #                 'Q': [17, 18, 19, 20], 
    #                 'R': [21, 22, 23, 24], 
    #                 'S': [25, 26, 27, 28], 
    #                 'T': [29, 30, 31, 32], 
    #                 'U': [1, 2, 3, 4], 
    #                 'V': [5, 6, 7, 8], 
    #                 'W': [9, 10, 11, 12], 
    #                 'X': [13, 14, 15, 16]}
    cultivar_dict = {'rep1':{'L1': [13, 14, 15, 16],
                              'L2': [29, 30, 31, 32],
                              'F1': [45, 46, 47, 48],
                              'F2': [61, 62, 63, 64],
                              'W1': [77, 78, 79, 80],
                              'W2': [93, 94, 95, 96]},
                      'rep2':{'L1': [41, 42, 43, 44],
                              'L2': [73, 74, 75, 76],
                              'F1': [89, 90, 91, 92],
                              'F2': [9, 10, 11, 12],
                              'W1': [25, 26, 27, 28],
                              'W2': [57, 58, 59, 60]},
                      'rep3':{'L1': [69, 70, 71, 72],
                              'L2': [37, 38, 39, 40],
                              'F1': [5, 6, 7, 8],
                              'F2': [53, 54, 55, 56],
                              'W1': [85, 86, 87, 88],
                              'W2': [21, 22, 23, 24]},
                      'rep4':{'L1': [17, 18, 19, 20],
                              'L2': [1, 2, 3, 4],
                              'F1': [65, 66, 67, 68],
                              'F2': [81, 82, 83, 84],
                              'W1': [33, 34, 35, 36],
                              'W2': [49, 50, 51, 52]}
                      }
    print(in_df.columns)
    ave_cultivar_combined_dict = {key: {c_key: [] for c_key in value.keys()} for key, value in cultivar_dict.items()}
    
    for col_name in in_df.columns:
        for rep_key, plot_dict in cultivar_dict.items():
            for cult_key, seg_nums in plot_dict.items():
                seg_idxs = np.asarray(seg_nums)-1
                seg_ave = np.around(np.average(in_df[col_name].iloc[seg_idxs]), decimals=3)
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
        
        #write the titles of the representatives horizontally. 
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