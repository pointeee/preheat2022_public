from skimage.segmentation import watershed
from skimage.morphology import local_minima

import copy
import numpy as np
from scipy import ndimage
from astropy import units as u

from field_util import Field


class ProtoClusterProber(Field):
# input should be delta_F
    def candidate_mask(self, crit):
        return self.field_data / self.field_data.std() < crit

    def minima_mask(self, tol=0.05):
        # g1mask = np.sqrt(ndimage.sobel(self.field_data, axis = 0)**2 
        #                + ndimage.sobel(self.field_data, axis = 1)**2
        #                + ndimage.sobel(self.field_data, axis = 2)**2) < tol
        # g2mask = ((ndimage.sobel(ndimage.sobel(self.field_data, axis = 0), axis = 0) > 0)
        #         * (ndimage.sobel(ndimage.sobel(self.field_data, axis = 1), axis = 1) > 0)
        #         * (ndimage.sobel(ndimage.sobel(self.field_data, axis = 2), axis = 2) > 0))
        # return g1mask*g2mask
        return local_minima(self.field_data)
    
    def mask_process(self, labels_, n_label, dist_allowed = 1*u.Mpc):
        '''
        dist_allowed requires the distances among representative points larger than the value.
        '''
        # mainly for the local minima
        # first, run a labeling to divide the grid
        # labels, n_label = ndimage.label(mask)
        # get individual masks
        labels = copy.deepcopy(labels_)

        label_list = [_ for _ in np.arange(1, n_label+1)]
        mask_center_list = []
        mask_label_list = []
        for label in label_list:
            indices = np.array(np.where(labels == label))
            com = self.index2coord(indices[0].mean(), indices[1].mean(), indices[2].mean())

            label_temp = label
            for i, prev_com in enumerate(mask_center_list):
                # dist
                x0, y0, z0 = prev_com
                x1, y1, z1 = com
                dist = np.sqrt((x1 -x0)**2 + (y1 -y0)**2 + (z1 -z0)**2)
                if dist < dist_allowed:
                    if mask_label_list[i] < label_temp:
                        labels[labels == label_temp] = mask_label_list[i]
                        label_temp = mask_label_list[i]
                    else:
                        labels[labels == mask_label_list[i]] = label_temp
                        label_to_edit = mask_label_list[i]
                        for j in range(len(mask_label_list)):
                            if mask_label_list[j] == label_to_edit:
                                mask_label_list[j] = label_temp
                else:
                    continue
            mask_center_list.append(com)
            mask_label_list.append(label_temp)

        mask_label_list = list(set(mask_label_list))
        mask_center_list = []
        for label in mask_label_list:
            indices = np.array(np.where(labels == label))
            com = self.index2coord(indices[0].mean(), indices[1].mean(), indices[2].mean())
            mask_center_list.append(com)
        # get the center of mass
        return labels, mask_center_list, mask_label_list

    def mask_to_list(self, mask):
        labels, n_label = ndimage.label(mask)
        mask_list = []
        for i in np.arange(1, n_label+1):
            mask_list.append(labels == i)
        return mask_list

    def deblend_cluster(self, cluster_mask_list, minima_mask):
        c_list = []
        for cluster in cluster_mask_list:
            if (cluster*minima_mask).sum() <= 1:
                indices = np.array(np.where(cluster))
                cluster_list = [cluster]
            else: 
                watershed_labels = watershed(self.field_data, mask = cluster)
                cluster_list = []
                for i in np.arange(1, watershed_labels.max()+1):
                    cluster = watershed_labels == i
                    cluster_list.append(cluster)
            c_list = c_list + cluster_list
        return c_list

    def find_cluster(self, crit=-2, tol=0.05, dist_allowed=1*u.Mpc):
        cmask = self.candidate_mask(crit=crit)
        mmask = self.minima_mask(tol=tol)

        cluster_mask_list = self.mask_to_list(cmask)
        # minima_mask_list = self.mask_to_list(mmask)

        cluster_mask_list = self.deblend_cluster(cluster_mask_list, mmask)
        # mask_new_list, center_new_list, minima_new_list = self.process_pairs(mask_list, minima_list)
        # center_list= np.array(center_list)
        # center_list = self.index2coord(center_list[:, 0], center_list[:, 1], center_list[:, 2])
        # center_list = np.array(center_list).T * u.Mpc

        labels = self.mask_to_labels(cluster_mask_list)
        n_labels = len(cluster_mask_list)
        labels, mask_center_list, mask_label_list = self.mask_process(labels, n_labels)
        cluster_mask_list = [labels==i for i in mask_label_list]

        return cluster_mask_list, mask_center_list
    

    def mask_to_labels(self, mask_list):
        labels = np.zeros_like(self.field_data)
        for i, mask in enumerate(mask_list):
            labels[mask] = i + 1
        return labels
            

    def is_in_mask(self, mask, coord):
        x, y, z = coord
        index = self.coord2index(x, y, z)
        return mask[index]

    def instant_view_pc(self):
        mask_list, center_list = self.find_cluster()
        np.random.seed(114514)
        g_list = []
        m_list = []

        for i, mask in enumerate(mask_list):
            color = np.random.random(size = 3)
            g, m = self.instant_view_region(mask, boundary=True, color=color)
            g_list = g_list + g
            m_list = m_list + m


            g, m = self.instant_view_sphere(radius=0.5*u.Mpc, coord=center_list[i], color=1-color)

            g_list = g_list + g
            m_list = m_list + m
        return g_list, m_list