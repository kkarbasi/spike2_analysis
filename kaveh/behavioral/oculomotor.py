"""
Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
import scipy.signal
from sklearn.cluster import KMeans

class target:
    """ Class for handling target data"""
    def __init__(self, vt, ht, dt, mode):
        """
        Object constructor
        m sklearn.cluster import KMeans
        vt: vertical target position signal
        ht: horizontal target position signal
        dt: sampling period
        """
        valid_modes = {'horizontal','vertical','2d'}
        if mode not in valid_modes:
            raise ValueError("Mode must be one of {}".format(valid_modes))
        if mode == 'horizontal':
            self.ht = ht
        if mode == 'vertical':
            self.vt = vt
        if mode == '2d':
            self.ht = ht
            self.vt = vt
        self.dt = dt
        self.mode = mode
    
    def _find_target_jumps(self):
        if self.mode == 'horizontal':
            return self._find_target_jumps_horizontal()
        if self.mode == 'vertical':
            return self._find_target_jumps_vertical()
        if self.mode == '2d':
            return self._find_target_jumps_2d()

    
    def _find_target_jumps_horizontal(self):
        """
        Finds the target jump indices in the input target horizontal position signal
        """
        # find target jumps
        ht_diff = np.abs(np.diff(self.ht))
        target_jump_indices = scipy.signal.find_peaks(ht_diff, prominence=200)[0]

        # remove detected target jumps that are sequential (less than 5 samples apart)
        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
                if tji - target_jump_indices[i] < 5:
                            to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]
        return target_jump_indices
        
    
    # TODO
    def _find_target_jumps_vertical(self):
        """
        Finds the target jump indices in the input target vertical position signal
        """
        # find target jumps
        vt_diff = np.abs(np.diff(self.vt))
        target_jump_indices = scipy.signal.find_peaks(vt_diff, prominence=200)[0]

        # remove detected target jumps that are likely noise related(less than 5 samples apart)
        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
                if tji - target_jump_indices[i] < 5:
                            to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]
        return target_jump_indices
    
    # TODO: try this in jupyter, also try detecting jumps using the 2d vector amp signal sqrt(x^2+y^2)
    def _find_target_jumps_2d(self):

        vt_diff = np.abs(np.diff(self.vt))
        target_jump_indices_v = scipy.signal.find_peaks(vt_diff, prominence=200)[0]
        ht_diff = np.abs(np.diff(self.ht))
        target_jump_indices_h = scipy.signal.find_peaks(ht_diff, prominence=200)[0]
        
        target_jump_indices = np.union1d(target_jump_indices_h, target_jump_indices_v)
        
        # remove detected target jumps that are likely noise related or
        # due to slight drift in detecting the same jump from horizontal and vertical signals
        # (less than 5 samples apart)
        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
                if tji - target_jump_indices[i] < 5:
                            to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]

        return target_jump_indices

        


        return 0

    def _find_jump_vector_amplitudes(self, num_clusters):
        if self.mode == 'horizontal':
            return self._find_jump_vector_amplitudes_h(num_clusters)
        if self.mode == 'vertical':
            return self._find_jump_vector_amplitudes_v(num_clusters)
        if self.mode == '2d':
            return self._find_jump_vector_amplitudes_2d(num_clusters)

    def _find_jump_vector_amplitudes_h(self, num_clusters):
        target_jump_indices = self._find_target_jumps()
        self.jump_vecs = []
        for tji in target_jump_indices:
                self.jump_vecs = self.jump_vecs + [self.ht[tji + 5] - self.ht[tji - 5]]
        #[hist, bin_edges] = np.histogram(jump_vecs, bins=np.arange(np.min(self.ht), np.max(self.ht), bin_size))
        #hist[hist < 10] = 0 # remove rare target jump vectors
        #return bin_edges[np.nonzero(hist)]
        self.jump_vecs = np.array(self.jump_vecs).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.jump_vecs)
        jump_amps = kmeans.cluster_centers_
        jump_amps = np.array([int(ja) for ja in jump_amps])
        return jump_amps

    def _find_jump_vector_amplitudes_v(self, num_clusters):
        target_jump_indices = self._find_target_jumps()
        self.jump_vecs = []
        for tji in target_jump_indices:
                self.jump_vecs = self.jump_vecs + [self.vt[tji + 5] - self.vt[tji - 5]]
        #[hist, bin_edges] = np.histogram(jump_vecs, bins=np.arange(np.min(self.ht), np.max(self.ht), bin_size))
        #hist[hist < 10] = 0 # remove rare target jump vectors
        #return bin_edges[np.nonzero(hist)]
        self.jump_vecs = np.array(self.jump_vecs).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.jump_vecs)
        jump_amps = kmeans.cluster_centers_
        jump_amps = np.array([int(ja) for ja in jump_amps])
        return jump_amps

    def _find_jump_vector_amplitudes_2d(self, num_clusters):
        return 0

    def _is_in_cluster(self, jump_vec, jump_amp, jump_tol):
        if self.mode == 'horizontal':
            if jump_vec < jump_amp + jump_tol and jump_vec >= jump_amp - jump_tol:
                return True
            else:
                return False
        if self.mode == 'vertical':
            # TODO
            return False
        if self.mode == '2d':
            # TODO
            return False

    def get_target_jumps(self, num_clusters = 3, jump_tol = 100):
        """
        Returns a dictionary containing the indices of the jumps to the found jump vectors.
        The jump vectors are found by detecting all jumps, then using kmeans with k=num_clusters
        (should be determined by the experimental setup, auto detection later), then assigning each
        jump to one cluster if the euclidean distance is less than jump_tol.
        """
        jump_amps = self._find_jump_vector_amplitudes(num_clusters)

        target_jumps_to = {}
        for ja in jump_amps:
                target_jumps_to[ja] = np.array([], dtype='int64')
        target_jump_indices = self._find_target_jumps()
        for i, tji in enumerate(target_jump_indices):
                #     jump_vec = ht.data[prange][tji + 5] - ht.data[prange][tji - 5]
                for ja in jump_amps:
                    if self._is_in_cluster(self.jump_vecs[i], ja, jump_tol):
                        target_jumps_to[ja] = np.concatenate((target_jumps_to[ja], [tji]))
        #target_jumps_to = {}
        #for ja in jump_amps:
        #        target_jumps_to[ja] = np.array([])
        #target_jump_indices = self._find_target_jumps()
        #for tji in target_jump_indices:
        #        jump_vec = self.ht[tji + 5] - self.ht[tji - 5]
        #        for ja in jump_amps:
        #            if jump_vec < ja + bin_size and jump_vec >= ja:
        #                target_jumps_to[ja] = np.concatenate((target_jumps_to[ja], [tji]))
        return target_jumps_to


