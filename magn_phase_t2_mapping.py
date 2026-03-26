#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:37:05 2023
Major Revision Sep 2025 

@authors: Difei Wang, Tony Stoecker
"""
import os
import time
from datetime import timedelta
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.optimize import least_squares

#parallel computing
import pymp

# nifti file support
import nibabel as nib

#fast epg's in python
import pyepg

# gradient echo signal with partial spoiling (GREPS) computed with EPG
def greps_signal_pyepg(T1, T2, alpha, delta_phi, TR, ABS_ERR_TOL = 1e-10 ):
    EPG = pyepg.PyEPG(1.0, T1, T2, TR)
    EPG.StepsToSS(alpha, delta_phi, ABS_ERR_TOL)
    signal = (EPG.GetReFa()+1.0j*EPG.GetImFa())*np.exp(-1.0j*np.deg2rad(EPG.GetPhase()))
    return ( np.abs(signal), np.angle(signal)+np.pi/2 )

# GREPS signal difference for least-squares fitting with scipy.optimize.least_squares
def greps_diff_signal_err_least_squares(x,fa,tr,dphi,signal):
    t1 = x[0]
    t2 = x[1]
    am = x[2] #amplitude modulation
    er = np.zeros((2*len(dphi),)) # error function of double length (real/imag values)
    n  = len(dphi)   #assume, sorted positive-only dphi values:  [dphi(0) ..., dph_(n-1), dph_n]  
    for i in range(n):
        a,p               = greps_signal_pyepg(t1,t2,fa,dphi[i],tr,1e-12)
        epgsig            = am * a * np.exp(-1.0j * p)
        er[2*i]           = np.real(epgsig-signal[i])
        er[2*i+1]         = np.imag(epgsig-signal[i])
    return er

# GREPS signal from pre-computed system-cube
def greps_signal_syscube(syscube, T1, T2, alpha):
    n = len(syscube.Phinc)
    s = np.zeros((2*n,))
    # interpolate system cube for each phase increment (real+imag)
    for i in range(2*n):
        V = np.squeeze(syscube.A[i,:,:,:])
        interp_syscube = rgi((syscube.T2_grid,syscube.T1_over_T2,syscube.FA), V, bounds_error=False,fill_value=100)
        s[i] = interp_syscube(np.array([T2,T1/T2,alpha]).T)
    return s


# GREPS signal difference with system-cube for least-squares fitting with scipy.optimize.least_squares
def greps_diff_signal_err_lsq_sc(x,syscube,fa,dphi,signal):
    t1 = x[0]
    t2 = x[1]
    am = x[2] #amplitude modulation
    gs = am * greps_signal_syscube(syscube,t1,t2,fa)    
    s = np.zeros((2*len(dphi),)) #  real/imag values from complex signal
    s[0::2]=np.real(signal)
    s[1::2]=np.imag(signal)
    return (s-gs)

# Helper class which loads or computes a 4D system cube of GREPS signals
class system_cube:
    def __init__(self, sc_file):
        self.sc_file = sc_file
        try:
            data = np.load(sc_file)
            self.A          = data['A']
            self.Phinc      = data['Phinc']
            self.FA         = data['FA']
            self.T2_grid    = data['T2_grid']
            self.T1_over_T2 = data['T1_over_T2']
        except:
            print('sc_file does not exist - use system_cube.calculate() to generate and save a new system cube')
            self.A          = []
            self.Phinc      = []
            self.FA         = []
            self.T2_grid    = []
            self.T1_over_T2 = []            

    def __str__(self):
        try:
            buf  = 'system cube file: ' + self.sc_file
            buf += '\n system cube shape = '+str(np.shape(self.A))
            buf += '\n Phinc =' +str(self.Phinc)
            buf += '\n T2_grid    (min,max,#): (%d,%d,%d)' % (np.min(self.T2_grid),np.max(self.T2_grid),len(self.T2_grid))
            buf += '\n T1_over_T2 (min,max,#): (%d,%d,%d)' % (np.min(self.T1_over_T2),np.max(self.T1_over_T2),len(self.T1_over_T2))
            buf += '\n FA         (min,max,#): (%d,%d,%d)' % (np.min(self.FA),np.max(self.FA),len(self.FA))
        except:
            buf = 'file does not exist - use system_cube.calculate() to generate and save a new system cube'
        return buf

    def calculate(self,n_proc,T2_grid,T1_over_T2,FA,Phinc,TR, ABS_ERR_TOL = 1e-10,VERBOSE=False):       
        start_time = time.time() 
        self.T2_grid    = T2_grid
        self.T1_over_T2 = T1_over_T2
        self.FA         = FA
        self.Phinc      = Phinc
        n_T2 = len(T2_grid); n_T1oT2 = len(T1_over_T2); n_FA = len(FA); n_Phinc = len(Phinc)
        if VERBOSE:
            print('compute system cube with dimensions')
            print('2*n_Phinc:',2*n_Phinc)
            print('n_T2     :',n_T2     )
            print('n_T1oT2  :',n_T1oT2  )
            print('n_FA     :',n_FA     )

        A = pymp.shared.array((2*n_Phinc,n_T2,n_T1oT2,n_FA), dtype='double')  
        progress   = pymp.shared.array((1,), dtype='uint8')

        with pymp.Parallel(n_proc) as p: 
            for j in p.range(0, n_T2):
                with p.lock:
                    progress[0] += 1 
                    if VERBOSE: p.print('progress: ',progress[0],' / ',n_T2)
                for k in np.arange(n_T1oT2):
                    for l in np.arange(n_FA):
                        for i in np.arange(n_Phinc):
                            #EPG
                                ampl , phase = greps_signal_pyepg(T1=T1_over_T2[k]*T2_grid[j], T2=T2_grid[j], alpha=FA[l], delta_phi=Phinc[i], TR = TR, ABS_ERR_TOL=ABS_ERR_TOL)
                                A[2*i  ][j][k][l] =   ampl * np.cos( phase ) #/ F0
                                A[2*i+1][j][k][l] = - ampl * np.sin( phase ) #/ F0
        self.A = A
        np.savez(self.sc_file, A=A, Phinc=Phinc,T2_grid=T2_grid,T1_over_T2=T1_over_T2,FA=FA)
        delta = time.time() - start_time
        if VERBOSE:
            print('\n\n  system cube stored to npz file (shape=',np.shape(A),')\n')
            print("->DONE in %2.f seconds (%s HMS)" %(delta, timedelta(seconds=delta)))    


# Main function to calculate T1, T2, and amplitude maps from GREPS magnitude and phase data, using either EPG or pre-computed system cube for signal modeling
def cal_T2T1AM(nprocs,magn_file, phas_file,mask_file, b1_file, TR, FA, phi, syscube=None,outputpath='.',outputbasename='result'):
    
    #% prepare input data
    d_phi = np.array(phi)
    print('PhaInc list:', d_phi)
    print('alpha=', FA)
    print('TR=', TR)    
    print('mask:', mask_file)
    #mask
    mask = nib.load(mask_file).get_fdata() >0.5
    print('========== Total number of voxels: ', np.sum(mask))
    
    #load B1 scale factor map
    print('Load B1 map', b1_file)
    B1scale = nib.load(b1_file).get_fdata()
    ALPHA = FA * B1scale[mask]/100.0 # [deg], 1D array
    
    # load magnitude images
    print('Load magnitude', magn_file)
    nii = nib.load(magn_file)
    magn_all = nii.get_fdata().astype(np.float32) #4D array
    print(magn_all.shape)
    
    # load phase images
    print('Load phase', phas_file)
    phas_all = nib.load(phas_file).get_fdata().astype(np.float32) #4D array
    print(phas_all.shape)
    
    T1_map = pymp.shared.array(mask.shape, dtype='double')
    T2_map = pymp.shared.array(mask.shape, dtype='double')
    Am_map = pymp.shared.array(mask.shape, dtype='double')
    
    # select corresponding phinc MAGNITUDE
    mag_image = np.zeros((len(d_phi), np.sum(mask)))
    for i in range(len(d_phi)):
        mag_image[i,:] = magn_all[:,:,:,i][mask]
    print(mag_image.shape)
        
    # select corresponding phinc PHASE
    phs_image = np.zeros((len(d_phi), np.sum(mask)))
    for i in np.arange(len(d_phi)):
        phs_image[i,:] = phas_all[:,:,:,i][mask]
    
    #complex signals
    complex_signal = mag_image*np.exp(-1j*phs_image)
    
    # % prepare T2 1D matrix
    T1_array = pymp.shared.array((np.sum(mask)), dtype='double')
    T2_array = pymp.shared.array((np.sum(mask)), dtype='double')
    Am_array = pymp.shared.array((np.sum(mask)), dtype='double')
    progress = pymp.shared.array((1,), dtype='uint32')

    # % iterate over voxels (True voxels in mask):
    t1_start = 1000.0
    t2_start = 80
    amp_scale_start = 5000
        
    x0   = [t1_start, t2_start, amp_scale_start]    #lsq start values  for T1, T2, and amplitude scaling
    bnds = ([1, 1, 1],[10000, 1000, 50000])         #lsq search bounds for T1, T2, and amplitude scaling
        
    print('num voxels=',np.sum(mask))    
    with pymp.Parallel(nprocs) as p: 
        for i in p.range(0, np.sum(mask)):
            progress[0] += 1
            if (np.mod(progress[0], 100) == 0):
                print('progress: ', progress[0], ' / ', np.sum(mask))
            #for i in p.range(1310, 1312):
            fa = ALPHA[i]
            sig = complex_signal[:, i]
            if syscube == None:
                args = (fa,TR,d_phi,sig)  
            else:
                args = (syscube,fa,d_phi,sig)  
                
            try:
                if syscube == None:
                    res = least_squares(greps_diff_signal_err_least_squares, x0, args=args, bounds=bnds, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10)
                else:
                    res = least_squares(greps_diff_signal_err_lsq_sc, x0, args=args, bounds=bnds, ftol = 1e-10, xtol = 1e-10, gtol = 1e-10)

                #print(res.x[0], res.x[1], res.x[2])
                T1_array[i] = res.x[0]
                T2_array[i] = res.x[1]
                Am_array[i] = res.x[2]
            except ValueError:
                print('i = %f' %i)
                T1_array[i] = 10000
                T2_array[i] = 10000
                Am_array[i] = 100000  
        
    # % prepare T1,T2,PD output nifti images
    T1_map[mask] = T1_array
    T2_map[mask] = T2_array
    Am_map[mask] = Am_array
    
        
    print('save as', outputpath, outputbasename)
    nib.save(nib.Nifti1Image(T1_map, nii.affine), os.path.abspath(os.path.join(outputpath,'T1_'+outputbasename+'.nii.gz' )))
    nib.save(nib.Nifti1Image(T2_map, nii.affine), os.path.abspath(os.path.join(outputpath,'T2_'+outputbasename+'.nii.gz' )))
    nib.save(nib.Nifti1Image(Am_map, nii.affine), os.path.abspath(os.path.join(outputpath,'Am_'+outputbasename+'.nii.gz' )))
    print('done')
    
    return (complex_signal,T1_array,T2_array,Am_array,ALPHA)