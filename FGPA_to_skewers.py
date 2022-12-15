from tkinter import Scale
import numpy as np
import pandas as pd
import copy

from field_util import Field

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM, z_at_value

class Mock_skewer(Field):
    '''
    This class implements the convertion from trans_field to skewer
    '''
    def add_skewer_data(self, skewer_pos_ori, skewer_pos, skewer_SN):
        self.skewer_pos_ori = skewer_pos_ori # position in original reconstruction
        self.skewer_pos = skewer_pos # position in COSTCO coordinates
        self.skewer_SN = skewer_SN
    
    def eval_raw_skewer(self):
        '''
        This function use transmission data to generate raw delta_f value for each skewer
        '''
        trans = self.field_eval(np.array(self.skewer_pos[:,0])*u.Mpc, 
                                np.array(self.skewer_pos[:,1])*u.Mpc,
                                np.array(self.skewer_pos[:,2])*u.Mpc) # this is trans value
        delta_F_raw = trans / np.nanmean(trans) - 1.
        return delta_F_raw

    def sigma_noise(self, conn_noise_func = lambda SN: 0.1 * np.ones_like(SN)):
        sigma = np.sqrt(1./self.skewer_SN**2 + conn_noise_func(self.skewer_SN)**2) / self.field_data.mean()
        return sigma


    def generate_skewer(self, seed=None):
        np.random.seed(seed)
        raw_skewer = self.eval_raw_skewer() # raw delta_F 
        sigma = self.sigma_noise().flatten() # full noise

        noise_skewer = raw_skewer + np.random.normal(scale=sigma) # get noisy skewer

        # export the result
        rec_input = np.vstack([self.skewer_pos_ori[:,0],
                               self.skewer_pos_ori[:,1],
                               self.skewer_pos_ori[:,2],
                               sigma, noise_skewer]).T
        rec_input_df = pd.DataFrame(rec_input, columns=['x', 'y', 'z', 'sigma_F_sim', 'delta_F_rec'])
        rec_input_df = rec_input_df.dropna()

        N_pix = len(rec_input_df)

        return rec_input_df, N_pix
        

