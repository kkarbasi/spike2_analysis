"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import math
import os
import copy
import numpy as np
import scipy.signal
import scipy.spatial
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
from matplotlib import pyplot as plot
from matplotlib import cm as cm
from toolbox.util import select, pca, find_first
from toolbox.plot import shaded_errorbar
import toolbox.smr.file
import h5py as h5


class SpikeDetector:
    """A class which encapsulates the properties associated with a spike detector"""

    def __init__(self, voltage, dt, max_events=5000):
        """Initialize the default arguments"""
        self.filter_order = 2
        self.high_pass_filter_cutoff = 1000
        self.low_pass_filter_cutoff = 10000
        self.zero_phase_filter = False
        self.num_pdf_bins = 100
        self.minimum_time_between_events = 0.001  # 1 ms
        self.max_alignment_jitter = self.minimum_time_between_events / 2
        self.principle_component_vaf = 0.75
        self.pre_window = 0.0005
        self.post_window = 0.0005
        self.max_neurons = 10  # Maximum number of neurons
        self.max_events = max_events
        # Store our parameters
        self.voltage = np.squeeze(np.array(voltage))
        self.dt = dt
        self.d_voltage = None
        self.triggers = None
        self.features = None
        self.weights = None
        self.average_waveforms = None
        self.num_spikes = 0
        self.mean_firing_rate = 0

    def run(self):
        # Run the pre-processing
        self._pre_process()
        self._compute_derivative()

        self._compute_thresholds()
        self.triggers = self._compute_crossings()
        self._align_triggers()

        if len(self.triggers) > self.max_events:
            indices = np.random.randint(0, len(self.triggers), self.max_events)
        else:
            indices = range(0, len(self.triggers))

        self._identify_features(indices=indices)
        self._compute_weights()
        self._determine_optimal_number_clusters(max_events=self.max_events, num_replicates=2)
        self.classify()

        # We now have a series of assignments, compute the average waveforms
        self.average_waveforms = self._compute_average_waveforms()

        # Compute statistics
        self.num_spikes = np.array([np.sum(self.assignments == i) for i in range(self.num_clusters)])
        self.mean_firing_rate = np.array([self.num_spikes[i] / (len(self.voltage) * self.dt) for i in range(self.num_clusters)])

    def _pre_process(self):
        """Perform pre-processing on a voltage timeseries"""
        # Generate the filter coefficients
        [b, a] = scipy.signal.filter_design.butter(self.filter_order,
                                [2 * self.dt * self.high_pass_filter_cutoff, 2 * self.dt * self.low_pass_filter_cutoff],
                                                   btype='bandpass')
        if self.zero_phase_filter:
            # Filter the signal using a zero-phase filter
            self.voltage = scipy.signal.lfilter(b, a, self.cs_voltage)  # Filter forwards
            self.voltage = np.flipud(self.cs_voltage)
            self.voltage = scipy.signal.lfilter(b, a, self.cs_voltage)  # Filter reverse
            self.voltage = np.flipud(self.cs_voltage)
        else:
            self.voltage = scipy.signal.lfilter(b, a, self.voltage)

    def _compute_derivative(self):
        """Compute the first difference"""
        #self.d_voltage = np.insert(np.diff(self.voltage) / self.dt, 0, 0)  # Ensure lengths remains the same
        self.d_voltage = scipy.signal.savgol_filter(self.voltage, 5, 2, 1, self.dt)

    def _compute_thresholds(self):
        """Computes thresholds on the original signal and the derivative"""
        self.voltage_pdf = self._calculate_signal_pdf(self.voltage)
        self.voltage_thresholds = self._calculate_fwhm(self.voltage)

        self.d_voltage_pdf = self._calculate_signal_pdf(self.d_voltage)
        self.d_voltage_thresholds = self._calculate_fwhm(self.d_voltage)

    def _compute_crossings(self):
        # Find indices that exceed our thresholds
        voltage_events = ((self.voltage < self.voltage_thresholds[0]) | (self.voltage > self.voltage_thresholds[1]))
        dvoltage_events = ((self.d_voltage < self.d_voltage_thresholds[0]) | (self.d_voltage > self.d_voltage_thresholds[1]))
        # Ensure that there is a differential event within the necessary window
        events = np.zeros(len(voltage_events), dtype=voltage_events.dtype)
        shifts = int(np.round(self.minimum_time_between_events / self.dt / 2))
        shifts = range(-shifts, shifts)
        for shift in shifts:
            shifted_dvoltage_events = copy.copy(dvoltage_events)
            if shift < 0:
                shifted_dvoltage_events = shifted_dvoltage_events[-shift:]

                shifted_dvoltage_events = np.insert(shifted_dvoltage_events, len(shifted_dvoltage_events), np.zeros(-shift))
            elif shift > 0:
                shifted_dvoltage_events = shifted_dvoltage_events[:(-shift)]
                shifted_dvoltage_events = np.insert(shifted_dvoltage_events, 1, np.zeros(shift))
            events = events | (voltage_events & shifted_dvoltage_events)

        # Ensure that there is a minimum time between events
        events = np.insert(events, [0, len(events)], [0, 0])
        events = events.astype(int)
        events = np.diff(events)
        t_on = np.where(events == 1)[0]
        t_off = np.where(events == -1)[0]

        # Ensure they are the same size
        length = min([len(t_on), len(t_off)])
        t_on = t_on[:length]
        t_off = t_off[:length]

        # Find the peak for each window
        peaks = np.zeros(len(t_on), dtype='uint32')
        for i in range(len(t_on)):
            peaks[i] = np.argmax(np.abs(self.voltage[t_on[i]:t_off[i]])) + t_on[i]

        # TODO: Ensure a minimum time between peaks
        remove = np.zeros(len(peaks), dtype=bool)
        for i in range(0, len(peaks) - 1):
            if (peaks[i+1] - peaks[i]) * self.dt < self.minimum_time_between_events:
                remove[i+1] = 1

        # Remove any spikes for which we do not have a full window
        for i in range(len(peaks)):
            if peaks[i] - np.round(self.pre_window / self.dt) < 0:
                remove[i] = 1
            else:
                break
        for i in range(len(peaks) - 1, 0, -1):
            if peaks[i] + np.round(self.post_window / self.dt) >= len(self.voltage):
                remove[i] = 1
            else:
                break
        peaks = select(peaks, ~remove)
        return peaks

    def _align_triggers(self):
        """Align triggers to the maximum value of the first derivative"""
        max_jitter_indices = int(np.round(self.max_alignment_jitter / self.dt))
        remove = np.zeros(len(self.triggers), dtype=bool)
        for i in range(0, len(self.triggers)):
            window_start = int(max([0, self.triggers[i] - max_jitter_indices]))
            window_stop = int(min([len(self.voltage), self.triggers[i] + max_jitter_indices]))

            # Find the maximum value of the first derivative within this window
            self.triggers[i] = np.argmax((self.d_voltage[window_start:window_stop])) + \
                               self.triggers[i] - max_jitter_indices
            if self.triggers[i] - np.ceil(self.pre_window / self.dt) < 0 or \
              self.triggers[i] + np.ceil(self.post_window / self.dt) > len(self.voltage):
                remove[i] = True
        self.triggers = self.triggers[~np.array(remove)]

    def _identify_features(self, indices=None):
        # Create a list of waveforms, concatenated
        start_indices = int(np.round(self.pre_window / self.dt))
        stop_indices = int(np.round(self.post_window / self.dt))
        if indices is None:
            indices = np.ones(len(self.triggers))
        waveforms = np.zeros((len(indices), start_indices + stop_indices), dtype=self.voltage.dtype)
        for i in range(len(indices)):
            waveforms[i, :] = self.voltage[self.triggers[indices[i]]-start_indices:self.triggers[indices[i]]+stop_indices]
        # Compute PCA on the components
        coeff, _, _, percent_explained = pca(waveforms)

        # Figure out how many features to use
        num_features = np.where(np.cumsum(percent_explained) >= self.principle_component_vaf * 100)[0][0] + 1
        print('Using {:d} features'.format(num_features))

        self.features = coeff[:, :num_features]

    def _compute_weights(self):
        start_indices = int(np.round(self.pre_window / self.dt))
        stop_indices = int(np.round(self.post_window / self.dt))

        self.weights = np.zeros((len(self.triggers), self.features.shape[1]))
        for i in range(len(self.triggers)):
            waveform = self.voltage[self.triggers[i] - start_indices:self.triggers[i] + stop_indices]
            self.weights[i, :] = np.dot(waveform, self.features)

        # Normalize each weight to a std of 1 (z-score)
        self.weights = self.weights - np.ones((len(self.weights), 1)) * np.mean(self.weights, axis=0)  # Remove mean
        self.weights = self.weights / (np.ones((len(self.weights), 1)) * np.std(self.weights, axis=0))  # Normalize

    def _determine_optimal_number_clusters(self, max_events=None, num_replicates=1):
        """Perform k-means clustering, use silhouette to determine optimal number of clusters"""
        silhouette = np.zeros(self.max_neurons-1)
        if max_events is None or max_events > len(self.triggers):
            max_events = None
            num_replicates = 1

        for replicates in range(0, num_replicates):
            for n_clusters in range(2, self.max_neurons + 1):
                if max_events is None:
                    indices = range(0, len(self.triggers))
                else:
                    indices = np.random.randint(0, len(self.triggers), max_events)
                clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, max_iter=1000, n_init=10)
                cluster_labels = clusterer.fit_predict(self.weights[indices, :])
                silhouette[n_clusters-2] += sklearn.metrics.silhouette_score(self.weights[indices, :], cluster_labels)

        print(silhouette)
        # Find the largest silhouette
        max_silhouette = np.max(silhouette)
        # Find the smallest number of clusters that is within  principle_component_vaf  of this silhouette
        self.num_clusters = np.where(silhouette > self.principle_component_vaf * max_silhouette)[0][0] + 2
        print('Assuming {:d} clusters'.format(self.num_clusters))

    def classify(self):
        # Use k nearest neighbors
        clusterer = sklearn.cluster.KMeans(n_clusters=self.num_clusters, max_iter=1000, n_init=50)
        self.assignments = clusterer.fit_predict(self.weights)

    def _compute_average_waveforms(self):
        """Average each of the assigned points"""
        start_indices = int(np.round(self.pre_window / self.dt))
        stop_indices = int(np.round(self.post_window / self.dt))

        average_waveforms = np.zeros((self.num_clusters, start_indices + stop_indices), dtype=self.voltage.dtype)
        for i in range(0, len(self.triggers)):
            average_waveforms[self.assignments[i], :] += self.voltage[
                              self.triggers[i] - start_indices:self.triggers[i] + stop_indices]
        # Normalize
        for i in range(0, self.num_clusters):
            average_waveforms[i, :] = average_waveforms[i, :] / np.sum(self.assignments == i)

        return average_waveforms

    def _compute_normalized_isi(self):
        """Assumes that we have assignments for all spikes"""
        isi = np.zeros(len(self.triggers))

        for i in range(0, len(self.triggers)):
            index = find_first(self.assignments == self.assignments[i], start=i, inclusive=False)
            if index is not None:
                isi[i] = (self.triggers[index] - self.triggers[i]) * self.dt
            else: # Wrap around
                index = find_first(self.assignments == self.assignments[i], start=0)
                isi[i] = ((len(self.voltage) - self.triggers[i]) + self.triggers[index]) * self.dt

        # Compute z-score
        isi = isi - np.mean(isi)
        isi = isi / np.std(isi)
        return isi

    def save(self, filename):
        """Save the results to a named HDF5 file"""
        file = h5.File(filename, 'w', libver='latest')
        self._save(file)
        file.close()

    def _save(self, file):
        """Save the results to an HDF5 file"""
        # Define a root group
        group = file.require_group("/")
        group.attrs['dt'] = self.dt
        group.attrs['filter_order'] = self.filter_order
        group.attrs['high_pass_filter_cutoff'] = self.high_pass_filter_cutoff
        group.attrs['low_pass_filter_cutoff'] = self.low_pass_filter_cutoff
        group.attrs['zero_phase_filter'] = self.zero_phase_filter
        group.attrs['max_neurons'] = self.max_neurons
        group.attrs['minimum_time_between_events'] = self.minimum_time_between_events
        group.attrs['max_alignment_jitter'] = self.max_alignment_jitter
        group.attrs['principle_component_vaf'] = self.principle_component_vaf
        group.attrs['pre_window'] = self.pre_window
        group.attrs['post_window'] = self.post_window
        group.attrs['max_events'] = self.max_events
        # Store our parameters
        group.attrs['num_clusters'] = self.num_clusters
        group.create_dataset("triggers", data=self.triggers)
        group.create_dataset("assignments", data=self.assignments)
        group.create_dataset("average_waveforms", data=self.average_waveforms)
        group.create_dataset("features", data=self.features)
        group.create_dataset("weights", data=self.weights)
        for i in range(self.num_clusters):
            group = file.require_group("/neuron_" + str(i))
            group.attrs['mean_firing_rate'] = len(self.triggers[self.assignments == i]) / (len(self.voltage) * self.dt)
            group.attrs['num_spikes'] = len(self.triggers[self.assignments == i])
            group.attrs['peak_to_peak_amplitude'] = np.max(self.average_waveforms[i]) - np.min(self.average_waveforms[i])
            group.create_dataset("spike_indices", data=self.triggers[self.assignments == i])


    def plot_thresholds(self):
        plot.figure()
        plot.bar(self.voltage_pdf[1], self.voltage_pdf[0])
        plot.axvline(self.voltage_thresholds[0], color='r')
        plot.axvline(self.voltage_thresholds[1], color='r')

        plot.figure()
        plot.bar(self.d_voltage_pdf[1], self.d_voltage_pdf[0])
        plot.axvline(self.d_voltage_thresholds[0], color='r')
        plot.axvline(self.d_voltage_thresholds[1], color='r')

    def plot_triggers(self, t=1):
        # Plot first 1 second of data
        indices = int(round(t / float(self.dt)))
        plot.figure()
        plot.plot(self.voltage[:indices])
        plot.axhline(self.voltage_thresholds[0], color='r')
        plot.axhline(self.voltage_thresholds[1], color='r')
        plot.scatter(self.triggers[self.triggers <= indices],
                     self.voltage[self.triggers[self.triggers < indices]], color='r')

        plot.figure()
        plot.plot(self.d_voltage[:indices])
        plot.axhline(self.d_voltage_thresholds[0], color='r')
        plot.axhline(self.d_voltage_thresholds[1], color='r')
        plot.scatter(self.triggers[self.triggers <= indices],
                     self.d_voltage[self.triggers[self.triggers < indices]], color='r')

    def plot_aligned_waveforms(self, t=0.5):
        indices = int(np.round(t / float(self.dt)))
        start_indices = int(np.round(self.pre_window / self.dt))
        stop_indices = int(np.round(self.post_window / self.dt))
        plot.figure()
        i = 0
        while (i < len(self.triggers) and self.triggers[i] < indices):
            window_start = int(max([0, self.triggers[i] - start_indices]))
            window_stop = int(min([len(self.voltage), self.triggers[i] + stop_indices]))
            plot.plot(self.voltage[window_start:window_stop])
            i += 1

    def plot_assigned_waveforms(self, pre_time=0.002, post_time=0.002, max_spikes=2000):
        plot.figure()
        start_indices = int(np.round(pre_time / self.dt))
        stop_indices = int(np.round(post_time / self.dt))
        if len(self.triggers) < max_spikes:
            max_spikes = len(self.triggers)
            indices = range(0, len(self.triggers))
        else:
            indices = np.random.randint(0, len(self.triggers), max_spikes)

        waveforms = np.zeros((max_spikes, start_indices + stop_indices), dtype=self.voltage.dtype) * np.nan
        for i in range(0, max_spikes):
            if self.triggers[indices[i]] - start_indices < 0 or self.triggers[indices[i]] + stop_indices > len(self.voltage):
                continue
            waveforms[i, :] = self.voltage[self.triggers[indices[i]] - start_indices:self.triggers[indices[i]] + stop_indices]

        for i in range(0, self.num_clusters):
            shaded_errorbar(range(0, waveforms.shape[1]),
                            np.nanmean(waveforms[self.assignments[indices] == i, :], axis=0),
                            np.nanstd(waveforms[self.assignments[indices] == i, :], axis=0))

    def plot_features(self,):
        for i in range(0, self.weights.shape[1] - 1):
            fig = plot.figure()
            ax = fig.add_subplot(111)
            if not hasattr(self, 'assignments'):
                plot.scatter(self.weights[:, i], self.weights[:, i+1], color='r')
            else:
                colors = cm.rainbow(np.linspace(0, 1, self.num_clusters))
                for j in range(0, self.num_clusters):
                    ax.scatter(self.weights[self.assignments == j, i], self.weights[self.assignments == j, i + 1], color=colors[j])

    @classmethod
    def _calculate_signal_pdf(cls, x, num_bins=100):
        """Calculates the PDF of the signal"""
        # Compute the bin centers (includes one extra point for np.histogram)
        centers = np.linspace(np.min(x), np.max(x), num_bins + 1)

        # Compute the pdf
        [pdf, _] = np.histogram(x, centers)

        # Remove the extra value from centers
        centers = centers[:-1]

        # Return the pdf and the bin centers
        return pdf, centers

    @classmethod
    def _calculate_fwhm(cls, voltage, scaling=1):
        """Calculates the full width at half maximum for the pdf"""
        # Determine the parameters from the pdf
        m = np.mean(voltage)
        s = np.std(voltage)

        # Find the full width at half max
        fwhm = (2 * math.sqrt(2 * math.log(2)) * s) * scaling
        return (m - fwhm, m + fwhm)


class ComplexSpikeDetector(SpikeDetector):

    def __init__(self, voltage, dt, max_events=5000):
        self.cs_voltage = copy.copy(voltage)
        super(ComplexSpikeDetector, self).__init__(voltage, dt, max_events)
        self.cs_mad_threshold = 10  # Number of MAD's (median absolute deviations) to consider a CS
        self.use_dt_cutoff = True
        self.cs_filter_order = 2
        self.cs_high_pass_filter_cutoff = 400
        self.cs_low_pass_filter_cutoff = 1000
        self.remove_ss_width = 1e-3
        self.cs_blanking_period = 10e-3
        self.cs_max_alignment_jitter = 2.5e-3
        self.cs_post_window = 5e-3
        self.cs_pre_window = 5e-3

    def run(self):
        super(ComplexSpikeDetector, self).run()

        # Preprocess for complex spikes
        self._cs_preprocess()

        # Z-score the voltage
        self._identify_cs_times()
        self._align_cs_triggers()
        self._remove_ss_triggers()

        # Compute the mean CS response
        self.average_cs_waveform = self._compute_average_cs_waveform()

    def _cs_preprocess(self):
        if self.use_dt_cutoff:
            [b, a] = scipy.signal.filter_design.butter(self.cs_filter_order,
                                                       [2 * self.dt * self.cs_high_pass_filter_cutoff,
                                                        2 * self.dt * self.cs_low_pass_filter_cutoff],
                                                       btype='bandpass')
        else:
            [b, a] = scipy.signal.filter_design.butter(self.cs_filter_order,
                                                       [2 * 1/100000 * self.cs_high_pass_filter_cutoff,
                                                        2 * 1/100000 * self.cs_low_pass_filter_cutoff],
                                                       btype='bandpass')
        if self.zero_phase_filter:
            # Filter the signal using a zero-phase filter
            self.cs_voltage = scipy.signal.lfilter(b, a, self.cs_voltage)  # Filter forwards
            self.cs_voltage = np.flipud(self.cs_voltage)
            self.cs_voltage = scipy.signal.lfilter(b, a, self.cs_voltage)  # Filter reverse
            self.cs_voltage = np.flipud(self.cs_voltage)
        else:
            self.cs_voltage = scipy.signal.lfilter(b, a, self.cs_voltage)

    def _identify_cs_times(self):
        # Convert the data to MAD's (assume an underlying gaussian distribution)
        med = np.median(self.cs_voltage)
        self.cs_voltage = np.abs(self.cs_voltage - med)
        mad = 1.4826 * np.median(self.cs_voltage)
        self.cs_voltage = self.cs_voltage / mad

        self.cs_mad_threshold = np.max(self.cs_voltage) * 0.75

        I = np.where((self.cs_voltage > self.cs_mad_threshold))[0]
        if I is not None and len(I) > 0:
            remove = np.insert(np.diff(I) <= int(self.cs_blanking_period / self.dt), 0, False)
            triggers = I[np.array(remove) == False]
        else:
            triggers = []
        print(triggers)
        self.cs_triggers = triggers

    def _remove_ss_triggers(self):
        width = np.ceil(self.remove_ss_width / self.dt / 2)
        for i in range(len(self.cs_triggers)):
            remove = (self.triggers > self.cs_triggers[i] - width) & (self.triggers < self.cs_triggers[i] + width)
            self.triggers = self.triggers[~np.array(remove)]
            self.assignments = self.assignments[~np.array(remove)]
            self.weights = self.weights[~np.array(remove), :]

    def _align_cs_triggers(self):
        """Align triggers to the maximum value of the first derivative"""
        max_jitter_indices = int(np.round(self.cs_max_alignment_jitter / self.dt))
        for i in range(0, len(self.cs_triggers)):
            window_start = int(max([0, self.cs_triggers[i] - max_jitter_indices]))
            window_stop = int(min([len(self.voltage), self.cs_triggers[i] + max_jitter_indices]))

            # Find the maximum value of the first derivative within this window
            self.cs_triggers[i] = np.argmax((self.d_voltage[window_start:window_stop])) + \
                               self.cs_triggers[i] - max_jitter_indices

    def _compute_average_cs_waveform(self):
        """Average each of the assigned points"""
        start_indices = int(np.round(self.cs_pre_window / self.dt))
        stop_indices = int(np.round(self.cs_post_window / self.dt))

        average_waveform = np.zeros(start_indices + stop_indices, dtype=self.cs_voltage.dtype)
        num_waveforms = 0
        for i in range(0, len(self.cs_triggers)):
            if self.cs_triggers[i] - start_indices < 0 or self.cs_triggers[i] + stop_indices > len(self.voltage):
                continue
            average_waveform += self.voltage[self.cs_triggers[i] - start_indices:self.cs_triggers[i] + stop_indices]
            num_waveforms += 1
        average_waveform /= num_waveforms

        return average_waveform

    def _save(self, file):
        """Overrides save in standard spike sorter"""
        super(ComplexSpikeDetector, self)._save(file)
        # Save complex spike parameters
        group = file.require_group("/")
        group.attrs['cs_mad_threshold'] = self.cs_mad_threshold
        group.attrs['use_dt_cutoff'] = self.use_dt_cutoff
        group.attrs['cs_filter_order'] = self.cs_filter_order
        group.attrs['cs_high_pass_filter_cutoff'] = self.cs_high_pass_filter_cutoff
        group.attrs['cs_low_pass_filter_cutoff'] = self.cs_low_pass_filter_cutoff
        group.attrs['cs_remove_ss_width'] = self.remove_ss_width
        group.attrs['cs_blanking_period'] = self.cs_blanking_period
        group.attrs['cs_max_alignment_jitter'] = self.cs_max_alignment_jitter
        group = file.require_group("/cs")
        group.attrs['num_spikes'] = len(self.cs_triggers)
        group.attrs['mean_firing_rate'] = len(self.cs_triggers) / (len(self.voltage) * self.dt)
        group.create_dataset("spike_indices", data=self.cs_triggers)

    def plot_cs_waveform(self, pre_time=0.005, post_time=0.005):
        fig = plot.figure()
        ax = fig.add_subplot(111)
        start_window = int(np.ceil(pre_time / self.dt))
        stop_window = int(np.ceil(post_time / self.dt))
        waveform = np.zeros(start_window + stop_window)
        for i in range(0, len(self.cs_triggers)):
            if self.cs_triggers[i] - start_window < 0 or self.cs_triggers[i] + stop_window > len(self.voltage):
                continue
            waveform = waveform + self.voltage[self.cs_triggers[i]-start_window:self.cs_triggers[i]+stop_window]
            if i < 50:
                ax.plot(self.voltage[self.cs_triggers[i]-start_window:self.cs_triggers[i]+stop_window])
        waveform = waveform / len(self.cs_triggers)

        ax.plot(waveform, linewidth=2, color='k')

    def plot_cs_triggers(self, t=1):
        # Plot first 1 second of data
        indices = round(t / float(self.dt))
        plot.figure()
        plot.plot(self.cs_voltage[:indices])
        plot.axhline(self.cs_mad_threshold, color='r')
        plot.scatter(self.cs_triggers[self.cs_triggers <= indices],
                     self.cs_voltage[self.cs_triggers[self.cs_triggers < indices]], color='r')

    def plot_cs_ss_isi(self, time_axis=np.arange(0, 50e-3, 1e-3)):
        for i in range(self.num_clusters):
            hist = np.zeros(len(time_axis))
            temp_triggers = self.triggers[self.assignments == i]
            num_events = 0
            for j in range(len(self.cs_triggers)):
                index = find_first(temp_triggers >= self.cs_triggers[j])
                if index is not None:
                    delta_t = (temp_triggers[index] - self.cs_triggers[j]) * self.dt
                    index = find_first(time_axis >= delta_t)
                    if index is not None:
                        hist[int(index)] += 1
                        num_events += 1
            hist /= num_events
            fig = plot.figure()
            ax = fig.add_subplot(111)
            ax.bar(time_axis, hist, width=np.mean(np.diff(time_axis)))
            ax.set_xlabel('Time for CS (s)')
            ax.set_ylabel('Probability')
            ax.set_title('Assignment = {:d}'.format(i))


class MultiFileSMRComplexDetector(ComplexSpikeDetector):

    def __init__(self, filenames, channel_number=0):
        """Load a series of filenames"""
        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.files = [toolbox.smr.file.File(filenames[i]) for i in range(len(filenames))]
        self.channels = [file.get_channel(channel_number) for file in self.files]

        dt = self.channels[0].dt
        voltage = np.hstack([channel.data for channel in self.channels])
        super(MultiFileSMRComplexDetector, self).__init__(voltage, dt)

    def save(self, postfix='spikes'):
        original_triggers = copy.copy(self.triggers)
        original_assignments = copy.copy(self.assignments)
        original_weights = copy.copy(self.weights)
        original_cs_triggers = copy.copy(self.cs_triggers)
        original_voltage = copy.copy(self.voltage)

        start_index = 0
        for i in range(len(self.filenames)):
            filename, _ = os.path.splitext(self.filenames[i])
            new_filename = filename + '_' + postfix + '.h5'
            file = h5.File(new_filename, 'w')
            end_index = start_index + len(self.channels[i].data)
            indices = np.array((original_triggers >= start_index) & (original_triggers < end_index))
            self.triggers = original_triggers[indices] - start_index
            self.assignments = original_assignments[indices]
            self.weights = original_weights[indices, :]
            self.voltage = original_voltage[start_index:end_index]
            indices = np.array((original_cs_triggers >= start_index) & (original_cs_triggers < end_index))
            self.cs_triggers = original_cs_triggers[indices] - start_index
            self._save(file)
            file.close()
            start_index = end_index


