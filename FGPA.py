import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.coordinates as coord
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import M_sun

import tensorflow as tf
from flowpm import utils as ut

import copy
import time
from sys import argv,exit,byteorder 


from field_util import Field

class FGPA(object):
    def __init__(self, fn, box_size = 512*u.Mpc, box_res = 256, origin = [3550*u.Mpc, -256*u.Mpc, -256*u.Mpc], 
                 tau_0 = 1, T_0 = 1, beta = 1.6, gamma = 1.6, from_npy = True):
        if from_npy:
            self.dark = self.read_from_npy(fn)
        else:
            self.header, self.gas, self.dark, self.star = self.tipsy_read(fn)
            self.auxilinary(box_size, box_res)
        self.set_origin(origin)
        self.set_FGPA_param(tau_0, T_0, beta, gamma)
        
    def set_FGPA_param(self, tau_0 = None, T_0 = None, beta = None, gamma = None):
        if tau_0 is not None:
            self.tau_0 = tau_0 # TBD
        if T_0 is not None:
            self.T_0   = T_0 # TBD
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        return 
        
    def set_origin(self, orig_pos):
        '''
        Set a offset value (move the origin point to the given [x0, y0, z0])
        add units!
        '''
        self.x0, self.y0, self.z0 =orig_pos
        return
        
    def auxilinary(self, box_size, box_res, N_particles = None):
        # settings of the simulation
        self.box_size = box_size # unit: Mpc
        self.box_res = box_res # unit: none
        self.res = box_size/box_res # unit: Mpc
        if N_particles is None:
            self.n_den = self.header['N'] / self.box_size**3
        else:
            self.n_den = N_particles / self.box_size**3
        #Do general cosmo things
        self.cosmo = FlatLambdaCDM(H0=100, Om0=0.315)
        self.mass_res = self.cosmo.critical_density0.to(u.M_sun/u.Mpc**3)*(self.box_size)**3/self.box_res**3*self.cosmo.Om(0.)
        
        #get speed of light in km/s
        self.ckms = 299792
        self.velkms = (self.box_size/8677.2079486362706)*self.ckms
        
    
    def read_from_npy(self, fn, costco_style = True):
        '''
        use this function to read reduced data, and recover the data to tipsy style.
        '''
        dark = np.load(fn)
        self.auxilinary(box_size = 512*u.Mpc, box_res = 256, N_particles = len(dark))

        dark = pd.DataFrame(dark,columns=dark.dtype.names)
        
        # basically what we do in the rev function
        dark['x'] = dark['x'] / self.box_size - 0.5 
        dark['y'] = dark['y'] / self.box_size - 0.5 
        dark['z'] = dark['z'] / self.box_size - 0.5
        dark['x'].units = None
        dark['y'].units = None
        dark['z'].units = None
        
        dark['vx'] = dark['vx'] / self.velkms
        dark['vy'] = dark['vy'] / self.velkms
        dark['vz'] = dark['vz'] / self.velkms
        dark['vx'].units = None
        dark['vy'].units = None
        dark['vz'].units = None
        
        if costco_style:
            dark['x'], dark['y'], dark['z'] = dark['y'], dark['x'], dark['z'] # ?
            dark['vx'], dark['vy'], dark['vz'] = dark['vy'], dark['vx'], dark['vz'] # dont forget the vel data!
            
        return dark
        
    
    def tipsy_read(self, fn, costco_style = True):
        tipsy = open(fn, 'rb')
        header_type = np.dtype([('time', '>f8'),('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'), ('pad', '>i4')])
        gas_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])
        dark_type = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('eps','>f4'), ('phi','>f4')])
        star_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
                            ('metals','>f4'), ('tform','>f4'), ('eps','>f4'), ('phi','>f4')])
        
        header = np.fromfile(tipsy,dtype=header_type,count=1)
        header = dict(zip(header_type.names,header[0]))
    
        gas  = np.fromfile(tipsy,dtype=gas_type,count=header['Ngas'])
        dark = np.fromfile(tipsy,dtype=dark_type,count=header['Ndark'])
        star = np.fromfile(tipsy,dtype=star_type,count=header['Nstar'])
        
        if byteorder == 'little':
            gas  =   gas.byteswap().newbyteorder('=')
            dark =   dark.byteswap().newbyteorder('=')
            star =   star.byteswap().newbyteorder('=')
        
        gas  = pd.DataFrame(gas,columns=gas.dtype.names)
        dark = pd.DataFrame(dark,columns=dark.dtype.names) # here is the raw data
        # in raw_data:
        # x - RA
        # y - DEC
        # z - red
        
        # what we want:
        # x - red
        # y - RA
        # z - DEC
        if costco_style:
            dark['x'], dark['y'], dark['z'] = dark['y'], dark['x'], dark['z'] # ?
            dark['vx'], dark['vy'], dark['vz'] = dark['vy'], dark['vx'], dark['vz'] # dont forget the vel data!
        star = pd.DataFrame(star,columns=star.dtype.names) 
        tipsy.close()
        return header, gas, dark, star
    
    def process_dark(self, dark = None):
        # for painting, keep the particles in [0, 512]^3 box
        # this function write in-situ results, so be careful
        # only use if you want to ensure the particles are in reasonable positions
        if dark is None:
            dark = copy.deepcopy(self.dark)
        
        dark['x'] = (dark['x']+0.5) * self.box_size + self.x0
        dark['y'] = (dark['y']+0.5) * self.box_size + self.y0
        dark['z'] = (dark['z']+0.5) * self.box_size + self.z0
        
        dark['x'].units = u.Mpc
        dark['y'].units = u.Mpc
        dark['z'].units = u.Mpc
        
        dark['vx'] = dark['vx'] * self.velkms
        dark['vy'] = dark['vy'] * self.velkms
        dark['vz'] = dark['vz'] * self.velkms
        
        dark['vx'].units = u.km * u.s**-1
        dark['vy'].units = u.km * u.s**-1
        dark['vz'].units = u.km * u.s**-1
    
        dark['mass'] = self.mass_res.value
        dark['mass'].units= M_sun
        
        return dark
    
    def process_dark_rev(self, dark_processed):
        '''
        recover the input field (to raw format)
        '''
        
        dark_processed['x'] = (dark_processed['x']-self.x0) / self.box_size - 0.5 
        dark_processed['y'] = (dark_processed['y']-self.y0) / self.box_size - 0.5 
        dark_processed['z'] = (dark_processed['z']-self.z0) / self.box_size - 0.5
        dark_processed['x'].units = None
        dark_processed['y'].units = None
        dark_processed['z'].units = None
        
        dark_processed['vx'] = dark_processed['vx'] / self.velkms
        dark_processed['vy'] = dark_processed['vy'] / self.velkms
        dark_processed['vz'] = dark_processed['vz'] / self.velkms
        dark_processed['vx'].units = None
        dark_processed['vy'].units = None
        dark_processed['vz'].units = None
        return dark_processed
        
    
    def particle_paint(self, nc, weight_col = None):
        '''
        nc: # of cells along any direction
        raw_part_data: *raw* particle data from the simuation (x, y, z, \\in [-0.5, 0.5])
        weight_col: pick one col in raw_data as weight; default value is 1 for all the particles 
        '''
        
        mesh = tf.zeros([1, nc, nc, nc], dtype = float)
        dark_raw = self.dark
    
        dark_pos = tf.convert_to_tensor([dark_raw['x'], dark_raw['y'], dark_raw['z']], dtype = float)
        dark_pos = tf.transpose((dark_pos + 0.5) * nc)
        dark_pos = tf.expand_dims(dark_pos, axis = 0) # [1, partN, 3]
        
        partN = dark_pos.shape[1]
        n_den = partN / nc**3
    
        if weight_col is None:
            weight = tf.ones([1, partN])
        else:
            weight = tf.convert_to_tensor(dark_raw[weight_col])
            weight = tf.expand_dims(weight, axis = 0)
        
        return ut.cic_paint(mesh, dark_pos, weight = weight) / n_den
    
    
    def particle_paint_clip_with_real_coord(self, real_coord_start, real_coord_end, dl, dark_raw = None, weight_col = None, smooth  = False):
        '''
        translate real space box to box in [-0.5, 0.5]^3 then run particle_paint_clip
        plz add units to all the parameters
        
        Issue: get things wrong here
        '''
        if dark_raw is None:
            dark_raw = self.dark
        
        rcs_x, rcs_y, rcs_z = real_coord_start
        rce_x, rce_y, rce_z = real_coord_end
        
        box_scale = round((self.box_size/dl).value)
        
        nc = np.round([((rce_x-rcs_x)/dl).value, 
                       ((rce_y-rcs_y)/dl).value, 
                       ((rce_z-rcs_z)/dl).value]).astype(int)
        offset = np.round([((rcs_x-self.x0)/dl).value, 
                           ((rcs_y-self.y0)/dl).value, 
                           ((rcs_z-self.z0)/dl).value]).astype(int)
        n_den = self.n_den * dl**3 #  global density per grid
        field_data = self.particle_paint_clip(nc, offset, box_scale, n_den, dark_raw = dark_raw, weight_col = weight_col)[0] # [nx, ny, nz] tensor
        field = Field(rcs_x, rcs_y, rcs_z, dl, field_data)
        if smooth:
            field.smooth(dl)
        return field
    
    def particle_paint_clip(self, nc, offset, box_scale, n_den, dark_raw = None, weight_col = None):
        '''
        nc: [nx, ny, nz]
            the shape/physical-scale of mesh.
        offset: [ox, oy, oz]
            [0,0,0] position of the mesh; default value  ox=0, oy=0, oz=0
        box_scale: the full scale of the simulation - box coord: [0, box_scale]^3
        # raw_part_data: *raw* particle data from the simuation (x, y, z, \\in [-0.5, 0.5])
        weight_col: pick one col in raw_data as weight; default value is 1 for all the particles 
        
        this function cutouts a box [ox, ox+nx] * [oy, oy+ny] * [oz, oz+nz] with field values.
        auto periodical condition(?)
        '''
        if dark_raw is None:
            dark_raw = self.dark
        
        nx, ny, nz = nc
        ox, oy, oz = offset
        
        mesh = tf.zeros([1, nx, ny, nz], dtype = float)
                
        # remove particles out of the boundary
        dark_clip = dark_raw[(dark_raw['x']<=(ox+nx)/box_scale-0.5) & (dark_raw['x']>= ox/box_scale-0.5) &
                             (dark_raw['y']<=(oy+ny)/box_scale-0.5) & (dark_raw['y']>= oy/box_scale-0.5) &
                             (dark_raw['z']<=(oz+nz)/box_scale-0.5) & (dark_raw['z']>= oz/box_scale-0.5)]
    
        dark_pos = tf.convert_to_tensor([(dark_clip['x']+ 0.5)*box_scale - ox, 
                                         (dark_clip['y']+ 0.5)*box_scale - oy, 
                                         (dark_clip['z']+ 0.5)*box_scale - oz], dtype = float)
        
        
        assert (np.max(dark_pos[0]) <= nx) & (np.max(dark_pos[1]) <= ny) & (np.max(dark_pos[2]) <= nz), print(np.max(dark_pos[0]), np.max(dark_pos[1]), np.max(dark_pos[2]))
        assert (np.min(dark_pos[0]) >= 0 ) & (np.min(dark_pos[1]) >= 0 ) & (np.min(dark_pos[2]) >= 0 )
        dark_pos = tf.transpose(dark_pos)
        dark_pos = tf.expand_dims(dark_pos, axis = 0) # [1, partN, 3]
        
        partN = dark_pos.shape[1]
    
        if weight_col is None:
            weight = tf.ones([1, partN])
        else:
            weight = tf.convert_to_tensor(dark_clip[weight_col])
            weight = tf.expand_dims(weight, axis = 0)
        
        paint = ut.cic_paint(mesh, dark_pos, weight = weight) / n_den
        
        
        return paint
    
    def RSD_catalog(self, real_coord_start, real_coord_end, dl):
        '''
        return a particle catalog with RSDed pos
        '''
        opd_clip = self.particle_paint_clip_with_real_coord(real_coord_start, real_coord_end, dl)
        fvx_clip = self.particle_paint_clip_with_real_coord(real_coord_start, real_coord_end, dl, weight_col = 'vx')
        fvy_clip = self.particle_paint_clip_with_real_coord(real_coord_start, real_coord_end, dl, weight_col = 'vy')
        fvz_clip = self.particle_paint_clip_with_real_coord(real_coord_start, real_coord_end, dl, weight_col = 'vz')
        
        part_new_pos  = self.field_to_part_pos(opd_clip)
        
        # generate a new particle catalog
        part_new = pd.DataFrame(part_new_pos[0], columns = ['x', 'y', 'z'])

        opd_tensor = tf.expand_dims(opd_clip.field_data, axis = 0)
        fvx_tensor = tf.expand_dims(fvx_clip.field_data, axis = 0)
        fvy_tensor = tf.expand_dims(fvy_clip.field_data, axis = 0)
        fvz_tensor = tf.expand_dims(fvz_clip.field_data, axis = 0)
        
        # flowpm.cic_readout requires a compatible mesh and coord ([nx, ny, nz] grid - [0, n*]^3 coord)
        part_pos_grid = self.mesh_to_part_pos(tf.expand_dims(opd_clip.field_data, axis = 0))
        part_new['opd'] = ut.cic_readout(opd_tensor, part_pos_grid)[0]
        part_new['vx']  = ut.cic_readout(fvx_tensor, part_pos_grid)[0]
        part_new['vy']  = ut.cic_readout(fvy_tensor, part_pos_grid)[0]
        part_new['vz']  = ut.cic_readout(fvz_tensor, part_pos_grid)[0]
                
        # convert to real space
        part_new = self.process_dark(part_new)
        
        # RSD
        # TODO: consider the effect of yz distance
        part_new['red_real'] = self.z_from_dist(part_new['x'] * u.Mpc)
        part_new['red_rs'] = part_new['red_real'] + part_new['vx'] / self.ckms
        part_new['x_red'] = self.z_to_dist(part_new['red_rs'])
        return part_new
        
    def FGPA_eval(self, mesh_catalog):
        # calculate tau and T for each particle
        # TODO: find out characterized tau_0 & T_0
        mesh_catalog['tau'] = self.tau_0 * mesh_catalog['opd']**self.beta

        mesh_catalog['T']   = self.T_0 * mesh_catalog['opd']**self.gamma
        return mesh_catalog
        
    def raw_tau_map(self, real_coord_start, real_coord_end, dl):
        
        # generate the particle catalog
        part_new = self.RSD_catalog(real_coord_start, real_coord_end, dl)
        # add tau / T to the catalog
        part_new = self.FGPA_eval(part_new)
        # this one in the RS space
        part_new_new = pd.DataFrame(part_new[['y', 'z', 'vx', 'vy', 'vz', 'tau', 'T']], columns = ['y', 'z',  'vx', 'vy', 'vz', 'tau', 'T'])
        part_new_new['x'] = part_new['x_red'] # not sure why this doesn't work
        # recover the raw_field
        part_new_new = self.process_dark_rev(part_new_new)
        # paint the final result
        tau_field = self.particle_paint_clip_with_real_coord(real_coord_start, real_coord_end, dl, 
                                                        dark_raw = part_new_new, weight_col = 'tau', smooth = False)
        return tau_field
    
    def tau_map(self, real_coord_start, real_coord_end, dl, z_comp = 2.30, F_obs = 0.8447):
        # derive the A_norm by comparing the desired value in obs with calculation
        self.set_FGPA_param(tau_0=1)
        raw_tau_field = self.raw_tau_map(real_coord_start, real_coord_end, dl)
        # construct the clip coord
        # hardcode bad, but works here
        clip_start = copy.deepcopy(real_coord_start)
        clip_start[0] = self.cosmo.comoving_distance(z_comp - 0.05)
        clip_end = copy.deepcopy(real_coord_end)
        clip_end[0] = self.cosmo.comoving_distance(z_comp + 0.05)
        
        test_tau_field = raw_tau_field.clip_with_coord(clip_start, clip_end)
        
        # no newton, just interp
        tau_list = test_tau_field.field_data
        tau_list = tau_list[~np.isnan(tau_list)]
        l = []
        for A in np.arange(0, 0.5, 0.001):
            l.append(np.mean(np.exp(-tau_list * A)))
        A_func = interp1d(l, np.arange(0, 0.5, 0.001))
        A_norm = A_func(F_obs)
        
        tau_field = copy.deepcopy(raw_tau_field)
        tau_field.field_data = tau_field.field_data * A_norm
        return tau_field
        
    
    def trans_map(self, real_coord_start, real_coord_end, dl):
        tau_field = self.tau_map(real_coord_start, real_coord_end, dl)
        trans_field = copy.deepcopy(tau_field)
        trans_field.field_data = np.exp(-trans_field.field_data)
        trans_field.field_data = trans_field.field_data / np.mean(trans_field.field_data)
        return trans_field
        
        
    def field_to_part_pos(self, field, tensor = True):
        '''
        
        '''
        nx, ny, nz = field.field_data.shape
        part_mesh = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
        part_mesh = np.reshape(part_mesh, [3, nx*ny*nz]).T.astype(float)
        part_mesh[:, 0] = (part_mesh[:, 0]*field.dl + field.x0 - self.x0) / self.box_size - 0.5 
        part_mesh[:, 1] = (part_mesh[:, 1]*field.dl + field.y0 - self.y0) / self.box_size - 0.5
        part_mesh[:, 2] = (part_mesh[:, 2]*field.dl + field.z0 - self.z0) / self.box_size - 0.5
        
        if tensor:
            part_mesh = tf.convert_to_tensor(part_mesh, dtype = float)
            part_mesh = tf.expand_dims(part_mesh, 0)
        return part_mesh

        
    def mesh_to_part_pos(self, mesh):
        '''
        Note this function is not complete: it converts the mesh into position [0, 0, 0] ... [nx, ny, nz] regardless the position
        Obselated
        '''
        _, nx, ny, nz = mesh.shape
        part_mesh = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
        part_mesh = np.reshape(part_mesh, [3, nx*ny*nz]).T.astype(float)
        
        part_mesh = tf.convert_to_tensor(part_mesh, dtype = float)
        part_mesh = tf.expand_dims(part_mesh, 0)
        return part_mesh
    
    def z_from_dist(self, distance):
        dummyred = np.linspace(0.,10.,10000)
        dummydist = self.cosmo.comoving_distance(dummyred)
        res = np.interp(distance,dummydist,dummyred)
        return (res)
    
    def z_to_dist(self, red):
        return self.cosmo.comoving_distance(red).value