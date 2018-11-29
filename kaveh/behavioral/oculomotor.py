"""
Copyright (c) 2018 Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
import scipy.signal

class target:
    """ Class for handling target data"""
    def __init__(self, vt, ht, dt, mode):
        """
        Object constructor
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
        Finds the target jump indices in the input target signal
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
        return 0
    
    # TODO
    def _find_target_jumps_2d(self):
        return 0

    def _find_jump_vector_amplitudes(self, bin_size):
        if self.mode == 'horizontal':
            return self._find_jump_vector_amplitudes_h( bin_size)
        if self.mode == 'vertical':
            return self._find_jump_vector_amplitudes_v( bin_size)
        if self.mode == '2d':
            return self._find_jump_vector_amplitudes_2d( bin_size)

    def _find_jump_vector_amplitudes_h(self, bin_size):
        target_jump_indices = self._find_target_jumps()
        jump_vecs = []
        for tji in target_jump_indices:
                jump_vecs = jump_vecs + [self.ht[tji + 5] - self.ht[tji - 5]]
        [hist, bin_edges] = np.histogram(jump_vecs, bins=np.arange(np.min(self.ht), np.max(self.ht), bin_size))
        hist[hist < 10] = 0 # remove rare target jump vectors
        return bin_edges[np.nonzero(hist)]

    def get_target_jumps(self, bin_size = 200):
        """
        Returns a dictionary containing the indices of the jumps to the found jump vectors.
        The jump vectors are found by detecting all the jumps, and finding the histogram of
        the amplitudes (in vertical and horizontal modes only for now). The bin size basically
        determines the resolution of detecting amplitudes
        """
        jump_amps = self._find_jump_vector_amplitudes(bin_size)
        target_jumps_to = {}
        for ja in jump_amps:
                target_jumps_to[ja] = np.array([])
        target_jump_indices = self._find_target_jumps()
        for tji in target_jump_indices:
                jump_vec = self.ht[tji + 5] - self.ht[tji - 5]
                for ja in jump_amps:
                    if jump_vec < ja + bin_size and jump_vec >= ja:
                        target_jumps_to[ja] = np.concatenate((target_jumps_to[ja], [tji]))
        return target_jumps_to


