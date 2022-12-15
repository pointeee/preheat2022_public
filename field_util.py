from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
import astropy.coordinates as coord

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LogNorm

from misc_func import sphere_kernel

import copy
import warnings

# from vis_util import o3d_vis, sphere_geo, points_geo, bounding_box_geo

class Field(object):
    '''
    This is the field class that enables basic analysis of given field.
    '''
    def __init__(self, x_0, y_0, z_0, dl, field_data):
        '''
        x_0, y_0. z_0 and dl have units of length

        Input
        "x_0", "y_0", "z_0" - coord of data grid [0, 0, 0]
        "dl" - the resolution of field data
        "field_data" - 3d array represents the field values
        '''

        self.x0 = x_0
        self.y0 = y_0
        self.z0 = z_0
        self.dl = dl
        self.field_data = field_data
        
    def coord2index(self, x, y, z):
        '''
        convert a given coordinate (list) into indices (list).
        To avoid OOB indexing, the indices are validated then output.
        '''
        ix = np.round(((x - self.x0)/self.dl).value)
        iy = np.round(((y - self.y0)/self.dl).value)
        iz = np.round(((z - self.z0)/self.dl).value)
        ix = self.valid_index_convert(ix, self.field_data.shape[0])
        iy = self.valid_index_convert(iy, self.field_data.shape[1])
        iz = self.valid_index_convert(iz, self.field_data.shape[2])
        return (ix,iy,iz)
    
    def valid_index_convert(self, ind, upper, lower = 0):
        '''
        function used to validate (and fix) the indices
        '''
        ind = np.array(ind)
        if (ind < lower).any() or (ind >= upper).any():
            warnings.warn("Warning: Index out of boundary")
        return ((ind * (ind >= lower) * (ind < upper)
                + lower * (lower > ind) 
                + (upper-1) * (upper <= ind))).astype(int)
    
    def valid_index_clip(self, ind_lower, ind_upper, upper, lower = 0):
        '''
        function used to validate indices used to clip the field      
        '''
        return (ind_lower >= lower) * (ind_upper <= upper) * (ind_lower < ind_upper)
    
    def index2coord(self, ix,iy,iz):
        '''
        convert a given indices (list) into coordinate (list).
        '''
        x = self.x0+ix*self.dl
        y = self.y0+iy*self.dl
        z = self.z0+iz*self.dl
        return (x,y,z)
    
    def smooth(self, sm_length):
        '''
        smooth the field with a gaussian filter.
        Note this function will perform in-situ modification on self.field_data,
        which makes it a non-pure function.

        Input -
        "sm_length" - a value with length unit. used as the radius of the gaussian filter

        Output - 
        smoothed field data. Note the data is also saved in the self.field_data
        '''
        sm_length_pix = (sm_length / self.dl).value
        self.field_data = ndimage.filters.gaussian_filter(self.field_data, sm_length_pix)
        return self.field_data

    def smooth_tophat(self, radius):
        '''
        smooth the field with a tophat filter.
        Note this function will perform in-situ modification on self.field_data,
        which makes it a non-pure function.
        '''
        radius_pix = (radius / self.dl).value
        kernel = sphere_kernel(radius_pix)
        self.field_data = ndimage.convolve(self.field_data, kernel, mode='wrap')
        return self.field_data

    def smooth_tophat_fft(self, radius):
        '''
        smooth the field with a tophat filter (with fft acceleration).
        Note this function will perform in-situ modification on self.field_data,
        which makes it a non-pure function.
        '''
        radius_pix = (radius / self.dl).value
        field_f = np.fft.fftn(self.field_data)       

        kernel = sphere_kernel(radius_pix)
        kernel_full = np.zeros_like(self.field_data)
        kernel_size = kernel.shape[0]
        s_x, s_y, s_z = np.shape(self.field_data)
        s_x, s_y, s_z = s_x//2 - kernel_size//2, s_y//2 - kernel_size//2, s_z//2 - kernel_size//2
        e_x, e_y, e_z = s_x+kernel_size, s_y+kernel_size, s_z+kernel_size
        kernel_full[s_x:e_x, s_y:e_y, s_z:e_z] = kernel
        kernel_full = np.fft.fftshift(kernel_full)
        kernel_full = kernel_full / np.sum(kernel_full)
        kernel_f = np.fft.fftn(kernel_full)

        filtered_f = field_f * kernel_f
        filtered = np.real(np.fft.ifftn(filtered_f))
        self.field_data = filtered
        return filtered

    
    def clip(self, index_range_lower, index_range_upper, check = True):
        '''
        lowest level clipping function.
        '''
        s = self.field_data.shape
        if check:
            assert self.valid_index_clip(index_range_lower[0], index_range_upper[0], s[0]), print('ix {} to {} not valid! (shape of data {})'.format(index_range_lower[0], index_range_upper[0], s))
            assert self.valid_index_clip(index_range_lower[1], index_range_upper[1], s[1]), print('iy {} to {} not valid! (shape of data {})'.format(index_range_lower[1], index_range_upper[1], s))
            assert self.valid_index_clip(index_range_lower[2], index_range_upper[2], s[2]), print('iz {} to {} not valid! (shape of data {})'.format(index_range_lower[2], index_range_upper[2], s))


        clip_field_data = self.field_data[
            index_range_lower[0]:index_range_upper[0],
            index_range_lower[1]:index_range_upper[1],
            index_range_lower[2]:index_range_upper[2],
        ]
        clip_x_0, clip_y_0, clip_z_0 = self.index2coord(
            index_range_lower[0], index_range_lower[1], index_range_lower[2])
        clip_field = Field(clip_x_0, clip_y_0, clip_z_0, 
                           self.dl, clip_field_data)
        return clip_field
    
    def clip_with_coord(self, coord_range_lower, coord_range_upper):
        '''
        Call clip with a coordinate range.
        '''
        index_lower = self.coord2index(
            coord_range_lower[0], coord_range_lower[1], coord_range_lower[2])
        index_upper = self.coord2index(
            coord_range_upper[0], coord_range_upper[1], coord_range_upper[2])
        return self.clip(index_lower, index_upper)

    def zoom_field(self, zoom):
        '''
        Zoom field. Returns the new field with altered resolution.
        '''
        new_field = copy.deepcopy(self)
        new_field.field_data = ndimage.zoom(self.field_data, zoom)
        new_field.dl = self.dl / zoom
        return new_field
    
    def mask_to_index(self, mask):
        '''
        Convert a mask of the field into a list of indices.
        '''
        assert mask.shape == self.field_data.shape
        s = mask.shape
        tot = np.prod(s)
        index_tot = np.arange(tot).reshape(s)
        index_list = []
        index_list.append(index_tot//(s[1]*s[2]))
        index_list.append((index_tot//s[2])%s[1])
        index_list.append(index_tot%s[2])
        index_list = np.array(index_list).transpose(1,2,3,0)
        return (index_list[mask].T) # to be agree with other functions
    
    def field_eval(self, x, y, z):
        '''
        return the field value at given position (x, y, z)
        use scipy.ndimage function so expect higher precison / slower speed
        '''
        return ndimage.map_coordinates(self.field_data, [(x-self.x0)/self.dl, 
                                          (y-self.y0)/self.dl, 
                                          (z-self.z0)/self.dl], order=1, cval=np.nan)
        
    
    def plot(self, title, tight_layout = True, vmin = None, vmax = None):
        '''
        Generate a quick view of the 3d field.
        '''
        if not tight_layout:
            plt.imshow(self.field_data.mean(axis = 0), origin = 'lower',
                       extent=[self.z0.value, self.z0.value+self.dl.value*self.field_data.shape[2], 
                              self.y0.value, self.y0.value+self.dl.value*self.field_data.shape[1]])
            plt.xlabel('$z / h^{-1} Mpc$')
            plt.ylabel('$y / h^{-1} Mpc$')
            plt.colorbar()
            plt.show()
            
            plt.imshow(self.field_data.mean(axis = 1), origin = 'lower', 
                       extent=[self.z0.value, self.z0.value+self.dl.value*self.field_data.shape[2], 
                               self.x0.value, self.x0.value+self.dl.value*self.field_data.shape[0]])
            plt.xlabel('$z / h^{-1} Mpc$')
            plt.ylabel('$x / h^{-1} Mpc$')
            plt.colorbar()
            plt.show()
    
            plt.imshow(self.field_data.mean(axis = 2), origin = 'lower',
                       extent=[self.y0.value, self.y0.value+self.dl.value*self.field_data.shape[1], 
                               self.x0.value, self.x0.value+self.dl.value*self.field_data.shape[0]])
            plt.xlabel('$y / h^{-1} Mpc$')
            plt.ylabel('$x / h^{-1} Mpc$')
            plt.colorbar()
            plt.show()
        else:
            assert self.field_data.shape[0] ==self.field_data.shape[1] and self.field_data.shape[1] ==self.field_data.shape[2]
            fig = plt.figure(figsize = (10, 3))
            grid = plt.GridSpec(3, 30)
            
            yz = plt.subplot(grid[0:3, 0:9])
            yz_plot = plt.imshow(self.field_data.mean(axis = 0), origin = 'lower',
                       extent=[self.z0.value, self.z0.value+self.dl.value*self.field_data.shape[2], 
                              self.y0.value, self.y0.value+self.dl.value*self.field_data.shape[1]])
            plt.clim(vmin, vmax)
            plt.xlabel('$z / h^{-1} Mpc$')
            plt.ylabel('$y / h^{-1} Mpc$')
            
            xz = plt.subplot(grid[0:3, 10:19])
            xz_plot = plt.imshow(self.field_data.mean(axis = 1), origin = 'lower', 
                       extent=[self.z0.value, self.z0.value+self.dl.value*self.field_data.shape[2], 
                               self.x0.value, self.x0.value+self.dl.value*self.field_data.shape[0]])
            plt.clim(vmin, vmax)
            plt.xlabel('$z / h^{-1} Mpc$')
            plt.ylabel('$x / h^{-1} Mpc$')
            
            xy = plt.subplot(grid[0:3, 20:29])
            xy_plot = plt.imshow(self.field_data.mean(axis = 2), origin = 'lower',
                       extent=[self.y0.value, self.y0.value+self.dl.value*self.field_data.shape[1], 
                               self.x0.value, self.x0.value+self.dl.value*self.field_data.shape[0]])
            plt.clim(vmin, vmax)
            plt.xlabel('$y / h^{-1} Mpc$')
            plt.ylabel('$x / h^{-1} Mpc$')
            
            cbar = plt.subplot(grid[0:3, 29:30])
            plt.suptitle(title)
            fig.colorbar(yz_plot, ax = [yz, xz, xy], cax = cbar)
            return fig

class CartCoord(object):
    '''
    This class is the bridge between the celestial coordinates and cartesian coordinates.
    To use this class, you need to feed (RA, Dec) 
    as the origin of the cartesian coordinates y=0 and z=0,
    and specify the cosmology used (this gives a natural definition of x=0 at z=0).

    If equipped with a field object, it can be used to manipulate the field.
    '''
    def __init__(self, RA, DE, cosmo):
        '''
        initialize the object with RA, DE, and cosmology.
        '''
        self.RA_origin = RA
        self.DE_origin = DE
        self.cosmo = cosmo
        self.field = None
    def orig_to_box(self,ra,dec,redshift):
        '''
        convert obs to grid pos;
        support the input as an array
        '''
        alphay = self.DE_origin # definition of rotation is with factor -1 in astropy
        alphaz = self.RA_origin - 180 # change with new data # Note this 180 deg
        rotationy = rotation_matrix(alphay, axis='y')
        rotationz = rotation_matrix(alphaz, axis='z')
        distance = self.cosmo.comoving_distance(redshift)
        coords_spherical = coord.SkyCoord((ra*u.degree),dec*u.degree,distance=distance)
        coords_cartesian = coords_spherical.cartesian
        coords_cartesian_new = coords_cartesian.transform(rotationz)
        coords_cartesian_newnew = coords_cartesian_new.transform(rotationy)
        coords_cartesian_newnew = coord.CartesianRepresentation(coords_cartesian_newnew.x*-1,
                                                                coords_cartesian_newnew.y,coords_cartesian_newnew.z)
        return (coords_cartesian_newnew.x,coords_cartesian_newnew.y,coords_cartesian_newnew.z)
    
    def original_pos(self,x,y,z):
        '''
        convert grid pos to obs
        support the input as an array
        '''
        coords_cartesian = coord.CartesianRepresentation(-(x)*u.Mpc,(y)*u.Mpc,(z)*u.Mpc)
        coords_spherical = coord.SkyCoord(coords_cartesian.represent_as(coord.SphericalRepresentation))
        coords_spherical = coord.SkyCoord(coords_spherical.ra,coords_spherical.dec,
                                          distance=coords_spherical.distance )
        coords_cartesian = coords_spherical.cartesian
        alphay = -self.DE_origin # definition of rotation is with factor -1 in astropy
        alphaz = -self.RA_origin + 180 # change with new data # Note this 180 deg
        rotationy = rotation_matrix(alphay, axis='y')
        rotationz = rotation_matrix(alphaz, axis='z')
        coords_cartesian_new = coords_cartesian.transform(rotationy)
        coords_cartesian_new = coords_cartesian_new.transform(rotationz)
        coords_spherical = coord.SkyCoord(coords_cartesian_new.represent_as(coord.SphericalRepresentation))
        
        def z_from_dist(distance):
            dummyred = np.linspace(0.,10.,10000)
            dummydist = self.cosmo.comoving_distance(dummyred)
            res = np.interp(distance,dummydist,dummyred)
            return (res)
    
        red =  z_from_dist(coords_spherical.distance)
        final = coord.SkyCoord(coords_spherical.ra,coords_spherical.dec,red)
        return (final)
    
    def add_field(self, field):
        '''
        This function load a field object on the coordinates.

        Input
        "field" - must be a field object
        '''
        self.field = field
        return
    
    def clip_field(self, coord_range_lower, coord_range_upper):
        '''
        clip the field with specified coordinate range.
        Actually call the clipping function of the field itself; so check its doc string for details.
        '''
        assert self.field is not None
        clip_field = self.field.clip_with_coord(coord_range_lower, coord_range_upper)
        clip_field_coord = copy.deepcopy(self)
        clip_field_coord.field = clip_field
        return clip_field_coord
    
    def map_data(self, comp_field, edge = False):
        '''
        this function estimate the value of self.field at given comp_field's lattice position.

        Input
        "comp_field" - must be a field object
        "edge" - True or False, 
                 if toggle True then the region out of self.field will be set to np.nan,
                 otherwise it will be the boundary value of self.field (due to the behavior of Field.coord2index)
        
        Output
        "map_field" - a field object, with size of comp_field, and values of self.field.
        '''
        comp_field_mask = np.ones_like(comp_field.field_data, dtype = bool)
        comp_field_index = comp_field.mask_to_index(comp_field_mask)
        comp_field_coord = comp_field.index2coord(
            comp_field_index[0], comp_field_index[1], comp_field_index[2]) #  = self.field 's coord
        field_index_x, field_index_y, field_index_z = self.field.coord2index(
            comp_field_coord[0], comp_field_coord[1], comp_field_coord[2])
        map_field_data = np.zeros_like(comp_field.field_data)
        s = self.field.field_data.shape
        if edge:
            for i in np.arange(comp_field.field_data.size):
                i_x, i_y, i_z = comp_field_index[0][i], comp_field_index[1][i], comp_field_index[2][i]
                ix0, iy0, iz0 = field_index_x[i], field_index_y[i], field_index_z[i]

                if ((ix0 == 0) or (ix0 == s[0]-1) 
                 or (iy0 == 0) or (iy0 == s[1]-1) 
                 or (iz0 == 0) or (iz0 == s[2]-1)):
                    map_field_data[i_x, i_y, i_z] = np.nan

                else:
                    map_field_data[i_x, i_y, i_z] = self.field.field_data[ix0, iy0, iz0]
        else:
            for i in np.arange(comp_field.field_data.size):
                i_x, i_y, i_z = comp_field_index[0][i], comp_field_index[1][i], comp_field_index[2][i]
                map_field_data[i_x, i_y, i_z] = self.field.field_data[
                    field_index_x[i], field_index_y[i], field_index_z[i]]
        map_field = copy.deepcopy(comp_field)
        map_field.field_data = map_field_data
        return map_field
