#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:43:01 2023

@author: wangd
"""
import numpy as np
import nibabel as nib
from scipy.optimize import minimize, least_squares
import equation_grep as eg

import time
from datetime import timedelta
import os
import glob
import pymp
import argparse
import sys


def cal_T2T1AM(magn_file, phas_file,mask_file, b1_file, TR, FA, phi, label=None, label_B1=None, outputdir=None):
    
    #%% prepare input data

    d_phi = np.array(phi)
    print('PhaInc list:', d_phi)
    
    label = label
    label_B1 = label_B1
    
    ALPHA = FA
    TR = TR
    print('alpha=', ALPHA)
    print('TR=', TR)
    
    
    print('mask:', mask_file)
    #mask
    mask = nib.load(mask_file).get_fdata() >0.5
    print('========== Total number of voxels: ', np.sum(mask))
    
    #load B1 scale factor map
    print('Load B1 map', b1_file)
    B1scale = nib.load(b1_file).get_fdata()
    ALPHA = ALPHA * B1scale[mask]/100.0 # [deg], 1D array
    
    # load magnitude images
    print('Load magnitude', magn_file)
    nii = nib.load(magn_file)
    magn_all = nii.get_fdata().astype(np.float32) #4D array
    print(magn_all.shape)
    
    # load phase images
    print('Load phase', phas_file)
    phas_all = nib.load(phas_file).get_fdata().astype(np.float32) #4D array
    print(phas_all.shape)
    
    
    
    T1_test = pymp.shared.array(mask.shape, dtype='double')
    T2_test = pymp.shared.array(mask.shape, dtype='double')
    Am_test = pymp.shared.array(mask.shape, dtype='double')
    
    
    # select corresponding phinc MAGNITUDE
    mag_image = np.zeros((len(d_phi), np.sum(mask)))
    for i in np.arange(len(d_phi)):
        #print(f_mag[i])
        mag_image[i,:] = magn_all[:,:,:,i][mask]
    print(mag_image.shape)
        
    # select corresponding phinc PHASE
    phs_image = np.zeros((len(d_phi), np.sum(mask)))
    for i in np.arange(len(d_phi)):
        #print(f_phs[i])
        phs_image[i,:] = phas_all[:,:,:,i][mask]
    
    #experimental complex signals
    complex_signal = mag_image*np.exp(-1.0j*phs_image)
    
    # %% prepare T2 1D matrix
    T1_array = pymp.shared.array((np.sum(mask)), dtype='double')
    T2_array = pymp.shared.array((np.sum(mask)), dtype='double')
    Am_array = pymp.shared.array((np.sum(mask)), dtype='double')
    progress = pymp.shared.array((1,), dtype='uint32')

    start_time = time.time()    
    # %% iterate over voxels (True voxels in mask):
    t1_start = 1000.0
    t2_start = 80
    amp_scale_start = 5000
        
    x0   = [t1_start, t2_start, amp_scale_start]    #lsq start values  for T1, T2, and amplitude scaling
    bnds = ([1, 1, 1],[10000, 1000, 50000])         #lsq search bounds for T1, T2, and amplitude scaling
        
    print('num voxels=',np.sum(mask))    
    with pymp.Parallel(16) as p: 
        for i in p.range(0, np.sum(mask)):
            progress[0] += 1
            if (np.mod(progress[0], 10) == 0):
                print('progress: ', progress[0], ' / ', np.sum(mask))
            #for i in p.range(1310, 1312):
            fa = ALPHA[i]
            sig = complex_signal[:, i]
            args = (fa,TR,d_phi,sig)  
                
            try:
                res = least_squares(eg.greps_diff_signal_err_least_squares, x0, args=args, bounds=bnds, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10)
                #print(res.x[0], res.x[1], res.x[2])
                T1_array[i] = res.x[0]
                T2_array[i] = res.x[1]
                Am_array[i] = res.x[2]
            except ValueError:
                print('i = %f' %i)
                T1_array[i] = 10000
                T2_array[i] = 10000
                Am_array[i] = 100000  
        
    # %% prepare T1,T2,PD output nifti images
    T1_test[mask] = T1_array
    
    T2_test[mask] = T2_array

    Am_test[mask] = Am_array
    
        
    print('==========================')
    delta = time.time() - start_time
    print("->DONE in %2.f seconds (%s HMS)" %(delta, timedelta(seconds=delta)))
    
    if label is None:
        outputbasename = ''
        
    else:
        outputbasename = '_'+label 
    
        
    if label_B1 is None:
        outputbasename = outputbasename
    else:
        outputbasename = outputbasename + '_' + label_B1
    
    
    if outputdir is None:
        outputpath = os.path.dirname(magn_file)
    else:
        outputpath = outputdir
        
        
    
    
    print('save as', outputpath, outputbasename)
    
    nib.save(nib.Nifti1Image(T1_test, nii.affine), os.path.abspath(os.path.join(outputpath,'T1_'+outputbasename+'.nii.gz' )))

    nib.save(nib.Nifti1Image(T2_test, nii.affine), os.path.abspath(os.path.join(outputpath,'T2_'+outputbasename+'.nii.gz' )))

    nib.save(nib.Nifti1Image(Am_test, nii.affine), os.path.abspath(os.path.join(outputpath,'Am_'+outputbasename+'.nii.gz' )))
    
    print('done')
    
    
    
    
    
    
    
    
        
        
        
    
    

def main():
    parser = argparse.ArgumentParser(
    description="Performs the script of T2-T1-PD fitting for number of phi test."
                )
    
    parser.add_argument('magn_file', type=str)
    parser.add_argument('phas_file', type=str)
    parser.add_argument('mask_file', type=str)
    parser.add_argument('b1_file', type=str)
    parser.add_argument('-tr', type=float)
    parser.add_argument('-fa', type=float)
    parser.add_argument('-phi', type=float, nargs='+')
    
    parser.add_argument('-label', type=str, default=None)
    parser.add_argument('-label_B1', type=str, default=None)
    parser.add_argument('-outputdir', type=str, default=None)
     
    
    args = parser.parse_args()
    
    
    cal_T2T1AM(args.magn_file, args.phas_file, args.mask_file, args.b1_file, args.tr, args.fa, args.phi,
                                                               args.label, args.label_B1, args.outputdir)

if __name__ == '__main__':
    sys.exit(main())        
    
    

 
