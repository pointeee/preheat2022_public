import numpy as np
import open3d as o3d
import pandas as pd
import os
from scipy import ndimage

from field_util import Field

# base function
def o3d_vis(geometry_list, material_list, name ="Visualizer", **kwargs):
    '''
    base function for o3d visualization. input are a list of geometry and a list of material, with the same length.
    '''
    print(kwargs)
    config_list = []
    for i, geo in enumerate(geometry_list):
        config_list.append({'name': str(i), 'geometry': geo, 'material': material_list[i]})
    o3d.visualization.draw(config_list, bg_color=[0, 0, 0, 1], show_skybox = False, **kwargs)

def points_geo(raw_point, color = [1, 1, 1]):
    '''
    Creating a set of points. Support adding color to the points.
    '''
    # create point cloud 
    pcd=o3d.open3d.geometry.PointCloud()
    # convert data to point cloud
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    # set color
    pcd.paint_uniform_color(color)
    return pcd

def sphere_geo(radius, coord, color= [1, 1, 1]):
    '''
    Creating a sphere with specified radius, position and color.
    '''
    sph = o3d.geometry.TriangleMesh.create_sphere(radius)
    sph.translate(coord, relative=False)
    sph.paint_uniform_color(color)
    return sph

def bounding_box_geo(coord_min, coord_max, color= [1, 1, 1]):
    '''
    Creating a bounding box with the coordinates specified.
    '''
    x0, y0, z0 = coord_min
    x1, y1, z1 = coord_max

    points = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
    ]
    print(points)
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def texture_opacity(alpha = 0.2):
    '''
    Generate the texture properties with alpha value.
    '''
    mat_box = o3d.visualization.rendering.MaterialRecord()
    mat_box.shader = 'defaultLitTransparency'
    # mat_box.shader = 'defaultLitSSR'
    mat_box.base_color = [1, 1, 1, alpha]
    #mat_box.base_roughness = 0.0
    #mat_box.base_reflectance = 0.0
    #mat_box.thickness = 1.0
    mat_box.transmission = 1.0
    #mat_box.absorption_distance = 10
    #mat_box.absorption_color = [0.5, 0.5, 0.5]
    return mat_box

class Field_v(Field):
    '''
    Add visualization features with o3d
    '''
    def instant_view_region(self, region, color=[1,1,1], boundary=True):
        assert self.field_data.shape == region.shape
        if boundary:
            mask = np.sqrt(ndimage.sobel(region, axis = 0, cval = 0)**2 
                         + ndimage.sobel(region, axis = 1, cval = 0)**2
                         + ndimage.sobel(region, axis = 2, cval = 0)**2) > 0 
        else:
            mask = region
        indices = np.array(np.where(mask))
        x, y, z = self.index2coord(indices[0], 
                                   indices[1], 
                                   indices[2])
        points = np.array([x.value, y.value, z.value]).T
        return [points_geo(points, color=color)], [None]

    def instant_view_extrema(self, tol=0.05, mask=None, color=[1,0,1], radius = 0):
        g1mask = np.sqrt(ndimage.sobel(self.field_data, axis = 0)**2 
                       + ndimage.sobel(self.field_data, axis = 1)**2
                       + ndimage.sobel(self.field_data, axis = 2)**2) < tol
        g2mask = ((ndimage.sobel(ndimage.sobel(self.field_data, axis = 0), axis = 0) > 0)
                * (ndimage.sobel(ndimage.sobel(self.field_data, axis = 1), axis = 1) > 0)
                * (ndimage.sobel(ndimage.sobel(self.field_data, axis = 2), axis = 2) > 0))
        if mask is None:
            indices = np.array(np.where(g1mask*g2mask))
        else:
            assert self.field_data.shape == mask.shape
            indices = np.array(np.where(g1mask*g2mask*mask))
        x, y, z = self.index2coord(indices[0], 
                                   indices[1], 
                                   indices[2])
        points = np.array([x.value, y.value, z.value]).T
        if radius == 0: # just plot the points
            return [points_geo(points, color=color)], [None]
        else: # plot sphere
            g = []
            m = []
            for coord in points:
                geo, mat = self.instant_view_sphere(radius*u.Mpc, coord*u.Mpc, color=color)
                g = g + geo
                m = m + mat
            return g, m


    def instant_view_sphere(self, radius, coord, color=[0,1,0]):
        x, y, z = coord 
        # no need to covert from coord to pix 
        # because we are doing visualization with the coordinates system
        return [sphere_geo(radius.value, [x.value, y.value, z.value], color=color)], [None]

    def instant_view(self, bounding_box=True, geometry_list=[], material_list=[], **kwargs):
        if bounding_box:
            s = self.field_data.shape
            x0, y0, z0 = self.x0.value, self.y0.value, self.z0.value
            x1, y1, z1 = (self.x0+self.dl*s[0]).value, (self.y0+self.dl*s[1]).value, (self.z0+self.dl*s[2]).value
            geometry_list.append(bounding_box_geo([x0, y0, z0], [x1, y1, z1]))
            material_list.append(None)
        o3d_vis(geometry_list, material_list, name ="Visualizer", **kwargs)