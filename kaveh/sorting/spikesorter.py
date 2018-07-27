"""
Copyright (c) 2018 Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.signal
from kaveh.plots import axvlines
from matplotlib import pyplot as plt

class SimpleSpikeSorter:
    """ Class that detects and sorts simple spikes"""
    def __init__(self, voltage, dt):
        """
        Object constructor
        """
        self.voltage = np.squeeze(np.array(voltage))
        self.dt = dt
        self.low_pass_filter_cutoff = 10000 #Hz
        self.high_pass_filter_cutoff = 1000 #Hz
        self.filter_order = 2
        self.num_gmm_components = 6
        self.gmm_cov_type = 'tied'
        self.pre_window = 0.0005 #s
        self.post_window = 0.0005 #s

    def run(self):
        self._pre_process()
        self._detect_spikes()
        self._align_spikes()        

    def _pre_process(self):
        """
        Pre-processing on the input voltage signal:
        Apply zero-phase linear filter
        """
        self.voltage_filtered = np.copy(self.voltage)
        [b, a] = scipy.signal.filter_design.butter(self.filter_order,
                        [2 * self.dt * self.high_pass_filter_cutoff, 2 * self.dt * self.low_pass_filter_cutoff],
                                               btype='bandpass')
        self.voltage_filtered = scipy.signal.lfilter(b, a, self.voltage_filtered)  # Filter forwards
        self.voltage_filtered = np.flipud(self.voltage_filtered)
        self.voltage_filtered = scipy.signal.lfilter(b, a, self.voltage_filtered)  # Filter reverse
        self.voltage_filtered = np.flipud(self.voltage_filtered)
        self.voltage_filtered = scipy.signal.savgol_filter(self.voltage_filtered, 5, 2, 1, self.dt)

    def _detect_spikes(self):
        """
        Preliminary spike detection using a Gaussian Mixture Model
        """
        gmm = GaussianMixture(self.num_gmm_components,
                covariance_type = 'tied').fit(self.voltage_filtered.reshape(-1,1))
        cluster_labels = gmm.predict(self.voltage_filtered.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(self.voltage_filtered.shape)
        spikes_cluster = np.argmax(gmm.means_)
        all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
        # Find peaks of each spike
        peak_times,_ = scipy.signal.find_peaks(self.voltage_filtered[all_spike_indices])
        self.spike_indices = all_spike_indices[peak_times]

    def _align_spikes(self, use_filtered=False, to_exclude=[]):
        """
        Aligns the spike waveforms (from pre_window to post_windows, aligned to peak times)
        Pass to_exclude if you want to exclude from adding to the align matrix
        """
        pre_index = int(np.round(self.pre_window / self.dt))
        post_index = int(np.round(self.post_window / self.dt))
        if use_filtered:
            self.aligned_spikes = np.array([self.voltage_filtered[i - pre_index : i + post_index ] 
                for i in self.spike_indices if i not in to_exclude and (i + post_index) <= self.voltage_filtered.size
                    and (i - pre_index) >= 0])
        else:
            self.aligned_spikes = np.array([self.voltage[i - pre_index : i + post_index ] 
                for i in self.spike_indices if i not in to_exclude and (i + post_index) <= self.voltage.size
                    and (i - pre_index) >= 0])
            

    # TODO
    def _choose_num_features(self, captured_variance=0.75):
        """
        Use the number of components that captures at least captured_variance of the spike waveforms
        """
        return 0

    def set_spike_window(self, pre_time, post_time):
        self.pre_window = pre_time
        self.post_window = post_time
        self._align_spikes()

    def plot_spike_waveforms_average(self, **kw):
        """
        Plots the average spike wavelets of the current dataset
        """
        spikes_avg = np.mean(self.aligned_spikes, axis = 0)
        spikes_std = np.std(self.aligned_spikes, axis = 0)/np.sqrt(self.aligned_spikes.shape[0])
        x = np.arange(0,self.aligned_spikes.shape[1])
        plt.figure()
        l = plt.plot(x, spikes_avg, **kw)
        plt.fill_between(x, spikes_avg - spikes_std, spikes_avg + spikes_std, color=l[0].get_color(), alpha=0.25)
        
    def plot_spike_peaks(self, use_filtered=False , figsize=(21,5)):
        """
        Plots the voltage signal, overlaid by spike peak times,
        overlaid by lines indicating the window around the spike peaks
        """
        if use_filtered:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage_filtered)
            plt.plot(self.spike_indices, self.voltage_filtered[self.spike_indices], '.r')
            axvlines(plt.gca(), self.spike_indices + int(np.round(self.post_window/self.dt)), color='g')
            axvlines(plt.gca(), self.spike_indices - int(np.round(self.pre_window/self.dt)), color='m')
        else:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage)
            plt.plot(self.spike_indices, self.voltage[self.spike_indices], '.r')
            axvlines(plt.gca(), self.spike_indices + int(np.round(self.post_window/self.dt)), color='g')
            axvlines(plt.gca(), self.spike_indices - int(np.round(self.pre_window/self.dt)), color='m')




        
        



