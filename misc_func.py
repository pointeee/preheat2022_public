
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LogNorm

def compare_fields(delta_F, delta_DM, R_sm, pc_meta):
    # from Metin 2019
    '''
    A function for comparing the fields of delta_F and delta_DM with hist2d.
    '''
    fig = plt.figure(figsize = (10, 7))
    
    bins_f = np.arange(-0.6,0.5, 0.01)
    bins_dm = np.arange(0., 8., 0.04)
    
    hist2d, edges_dm, edges_f = np.histogram2d(np.ndarray.flatten(delta_DM.field_data), np.ndarray.flatten(delta_F.field_data), 
                                bins=[bins_dm, bins_f],density=True)
    
    X, Y = np.meshgrid(edges_dm,  edges_f, indexing='ij')
    plt.pcolormesh(X,Y, hist2d, cmap='Greys',
                  norm=LogNorm(vmin=2e-3, vmax=100.))
    
    cbar = plt.colorbar()
    cbar.set_label('normalized density')
    
    XCon, YCon = np.meshgrid(edges_dm[0:-1]+(edges_dm[1]-edges_dm[0])/2 , 
                             edges_f[0:-1]+(edges_f[1]-edges_f[1])/2 , 
                             indexing='ij')
    # plt.contour(XCon,YCon, hist2d, levels = 3)
    
    plt.xlabel('$\\delta_{DM}$')
    plt.ylabel('$\\delta_{F}$')
    
    plt.title('$\\delta_{DM} - \\delta_{F}$ of ' 
              + '{}, \nRA: {}, DE: {} '.format(pc_meta['Name'], pc_meta['RA'], pc_meta['DE']) 
              + '$R_{sm}$ = ' + str(R_sm))
    return fig

def compare_fields_general(field_1, field_2, extent, ncell_1, ncell_2, vmin = 2e-3, vmax = 100, countour = True):
    # from Metin 2019
    '''
    extent = [x_1, y_1, x_2, y_2]
    '''
    fig = plt.figure(figsize = (10, 10))
    
    x_1, y_1, x_2, y_2 = extent
    
    bins_1 = np.linspace(x_1, x_2, ncell_1)
    bins_2 = np.linspace(y_1, y_2, ncell_2)
    
    hist2d, edges_1, edges_2 = np.histogram2d(np.ndarray.flatten(field_1.field_data), np.ndarray.flatten(field_2.field_data), 
                                bins=[bins_1, bins_2],density=True)
    
    X, Y = np.meshgrid(edges_1,  edges_2, indexing='ij')
    plt.pcolormesh(X,Y, hist2d, cmap='Greys',
                  norm=LogNorm(vmin=vmin, vmax=vmax))
    
    cbar = plt.colorbar()
    cbar.set_label('normalized density')
    
    XCon, YCon = np.meshgrid(edges_1[0:-1]+(edges_1[1]-edges_1[0])/2 , 
                             edges_2[0:-1]+(edges_2[1]-edges_2[1])/2 , 
                             indexing='ij')
    
    if countour:
        plt.contour(XCon,YCon, hist2d, levels = 5)
    return fig

def sphere_kernel(radius, normalize = True):
    size = int(radius)*2+1
    grid = np.array(np.meshgrid(np.arange(size), np.arange(size), np.arange(size)))
    kernel = ((grid - int(radius))**2).sum(axis=0) < int(radius)**2
    if normalize:
        kernel = kernel / kernel.sum()
    return kernel