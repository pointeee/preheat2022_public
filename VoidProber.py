from field_util import Field

import numpy as np
from astropy import units as u


class Ball(object):
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
    
    def __repr__(self):
        return ("<r = {0} ball at ({1}, {2}, {3})".format(self.radius, self.x, self.y, self.z))
        
    def __str__(self):
        return ("<r = {0} ball at ({1}, {2}, {3})".format(self.radius, self.x, self.y, self.z))

    def have_inter(self, other):
        assert type(other) == type(self), "TypeError: other must be ball object"
        return (self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2 < (self.radius + other.radius)**2

    def to_mask(self, f: Field):
        s = f.field_data.shape
        grid = np.array(np.meshgrid(np.arange(s[0]), np.arange(s[1]), np.arange(s[2]), indexing='ij'))
        grid_coord = f.index2coord(grid[0], grid[1], grid[2])
        mask_sph = ((grid_coord[0]- self.x)**2+ (grid_coord[1]- self.y)**2+(grid_coord[2]- self.z)**2) < self.radius**2
        return mask_sph

    def average_in_field(self, f: Field):
        mask = self.to_mask(f)
        return np.nanmean(f.field_data[mask])

class VoidDetect(Field):
    sparse = 2*u.Mpc
    radius_upper = 10*u.Mpc
    radius_lower = 2*u.Mpc
    tol = 0.1*u.Mpc
    def generate_seed(self, delta_void, sparse):
        zoom = self.dl / sparse
        new_field = self.zoom_field(zoom.value)
        v_mask = new_field.field_data > delta_void
        ix, iy, iz = np.where(v_mask)
        seed_list = np.array(new_field.index2coord(ix, iy, iz)).T*u.Mpc
        return seed_list

    def solve_radius(self, seed, delta_average, radius_upper, radius_lower, tol):
        x, y, z = seed
        field_temp = self.clip_with_coord([x-radius_upper, y-radius_upper, z-radius_upper], 
                                          [x+radius_upper, y+radius_upper, z+radius_upper])
        while abs(radius_upper-radius_lower) > tol:
            ball_upper = Ball(x, y, z, radius_upper)
            mean_upper = ball_upper.average_in_field(field_temp) # this should be lower than delta_average
            ball_lower = Ball(x, y, z, radius_lower)
            mean_lower = ball_lower.average_in_field(field_temp) # this should be higher than delta_average
            if not ((mean_upper < delta_average) and (mean_lower > delta_average)):
                return Ball(x, y, z, 0*u.Mpc)
            radius_new = (radius_lower + radius_upper) / 2
            ball_new = Ball(x, y, z, radius_new)
            mean_new = ball_new.average_in_field(field_temp)
            if mean_new < delta_average:
                radius_upper = radius_new
            else:
                radius_lower = radius_new
        return ball_new
            
    def void_list(self, seed_list, delta_average):
        for seed in seed_list:
            yield self.solve_radius(seed, delta_average, self.radius_upper, self.radius_lower, self.tol)

    def clean_void_list(self, v_list, radius_cut = 2*u.Mpc):
        cv_list = []
        for i, v in enumerate(v_list):
            cv_list = list(filter(None, cv_list))
            if v.radius < radius_cut:
                continue
            for i, u in enumerate(cv_list):
                if v.have_inter(u):
                    if u.radius <= v.radius:
                        cv_list[i] = None
                        continue
                    else:
                        break
                else:
                    continue
            if len(cv_list) == 0:
                cv_list.append(v)
                continue    
            if (v.radius < u.radius) and (v.have_inter(u)):
                continue
            else:
                cv_list.append(v)
        return cv_list

    def find_void(self, delta_void, delta_average):
        assert delta_void > delta_average, "ValueError: delta_average must be larger than delta_average"
        seed_list = self.generate_seed(delta_void, self.sparse)
        v_list = list(self.void_list(seed_list, delta_average)) # takes ~ 40 s
        cleaned_v_list = list(self.clean_void_list(v_list)) # takes ~ 40 s
        return cleaned_v_list