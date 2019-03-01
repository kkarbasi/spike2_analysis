#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:41:12 2019

@author: kkarbasi
"""
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import numpy as np
from matplotlib import pyplot as plt
from neo.io import Spike2IO

from kaveh.behavioral import oculomotor
from kaveh.sorting import spikesorter
from kaveh.toolbox import find_file
from kaveh.plots import axvlines
from smr import File
import os

f_names= [
          '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2008_Random/Kimo/K48/error_direction/K48_2_CSddirTuning.smr',
         '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2008_Random/Kimo/K69/error_direction/K69_1_DirTuning.smr',
         '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2008_Random/Kimo/K69/error_magnitude/K69_1_ErrorSize45degDir.smr',
         '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2008_Random/Kimo/K16/error_direction/K16_2_directionaltest.smr',
         '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2008_Random/Step/S38/error_direction/S38_1_directionaltuning.smr',
          '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2006/Oscar/O62/O62_1_FW5R_BW5L_A.smr',
          '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2006/Oscar/O62/O62_1_pre.smr',
         ]
cs_spiketrain_idx = [1, 1, 2, 0, 1, 2, 0]
#cs_spiketrain_idx = [ 0, 1, 2, 0]



#%%
f_id = 6
num_spike_train = cs_spiketrain_idx[f_id]


cs_path = '/mnt/data/temp/kaveh/auto_processed/' 
cf = find_file(os.path.split(f_names[f_id])[1] + '.pkl', cs_path)
with open(cf, 'rb') as input:
    try:
        sss = pickle.load(input, encoding='latin1')
    except TypeError:
        sss = pickle.load(input)


neo_reader = Spike2IO(filename=f_names[f_id])
neo_data = neo_reader.read()
data_block = neo_data[0]
seg = data_block.segments[0]

cs_indices = np.array([])
for i in range(num_spike_train+1):
    cs_indices = np.union1d(cs_indices, np.array(seg.spiketrains[i]))
cs_indices = np.int32(cs_indices/sss.dt)


def construct_y(spike_indices, cs_indices):
    from kaveh.toolbox import closest_argmin
    cs = closest_argmin(cs_indices, spike_indices)
    labeled = np.zeros(spike_indices.shape)
    labeled[cs] = 1.0
    
    return labeled


smr_content = File(f_names[f_id])
smr_content.read_channels()
voltage_chan = smr_content.get_channel(0)
sss.voltage = voltage_chan.data
spike_indices = sss.get_spike_indices()


#plt.figure(figsize=(15,5))
#
#plt.plot(sss.voltage, alpha = 0.3, zorder = 1)
#plt.eventplot(spike_indices, alpha = 0.5, linelengths=5000, color = 'r')
#plt.eventplot(cs_indices, alpha = 0.5, linelengths=10000, color = 'g')
#plt.show()
    
#%%
from kaveh.toolbox import closest_argmin
spike_indices_cs = closest_argmin(cs_indices, spike_indices)
spike_indices_cs = spike_indices[spike_indices_cs]

plt.figure()
plt.show(block=False)

for i, csi in enumerate(cs_indices):
    plt.cla()
    trace = sss.voltage[csi-1000: csi+1000]
    plt.plot(trace, alpha = 0.3, zorder = 1)
    plt.eventplot([csi - spike_indices_cs[i] + 1000], linelengths=np.max(trace)*2.1, color = 'g')
    plt.ylim((np.min(trace)*1.1, np.max(trace)*1.1))
    plt.title(i)
    plt.pause(0.05)
    plt.waitforbuttonpress()

        
    
#    plt.eventplot(spike_indices, alpha = 0.5, linelengths=5000, color = 'r')
#    plt.eventplot(cs_indices, alpha = 0.5, linelengths=10000, color = 'g')
#%%
#plt.figure()
##plt.show()
#    
#for i in range(10):
##    plt.cla()
#    plt.xlim((0,10))
#    plt.ylim((0,10))
#    plt.plot(range(0,i))
#    plt.show()
#    plt.waitforbuttonpress()    
#    
#   

#%%
for fn in f_names:
    smr_content = File(fn)
    smr_content.read_channels()
    voltage_chan = smr_content.get_channel(0)
    sss = spikesorter.SimpleSpikeSorter(voltage_chan.data, voltage_chan.dt)
    sss.run()
#    sss._detect_spikes_minibatch()
    with open(os.path.join('../data/spike_sorted_fixed_minibatch', os.path.basename(fn) + '.pkl'), 'wb') as output:
        pickle.dump(sss, output, pickle.HIGHEST_PROTOCOL)
        
    


#%%

sss = spikesorter.SimpleSpikeSorter(voltage_chan.data, voltage_chan.dt)
sss._pre_process()
sss._detect_spikes()

#%%
#
#spike_peaks = np.array([np.argmax(sss.voltage[si - int(0.0005/sss.dt): si + int(0.002/sss.dt)]) for si in sss.spike_indices])
#
#spike_indices = sss.spike_indices + spike_peaks - int(0.0005/sss.dt)

#%%

plt.figure(figsize=(15,5))
plt.plot(sss.voltage, alpha = 0.5, color='r')
plt.plot(sss.voltage_filtered*0.00009, alpha = 0.5, color='g')
plt.eventplot(sss.spike_indices, alpha = 0.5, linelengths=np.max(sss.voltage)*2, colors='y')
plt.show()

    
#%%
#fname = '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2010_5_types_of_saccades/Buckley/BuckleyUnit 2007.11.08/BuckleyUnit_2007.11.08_1736_Three.smr'
f_name = f_names[2]



smr_content = File(f_name)
smr_content.read_channels()
voltage_chan = smr_content.get_channel(0)
sss = spikesorter.SimpleSpikeSorter(voltage_chan.data, voltage_chan.dt)
sss._pre_process()
sss._detect_spikes_minibatch()


#74119648.0



#%%
f_name = f_names[0]
smr_content = File(f_name)
smr_content.read_channels()
voltage_chan = smr_content.get_channel(0)

#     cs_path = '/mnt/data/temp/kaveh/auto_processed/' 
# cs_path = '../data/auto_processed_spike_sorting/'
cs_path = '../data/spike_sorted_fixed_minibatch'
cf = find_file(os.path.split(f_name)[1] + '.pkl', cs_path)
with open(cf, 'rb') as input:
    sss = pickle.load(input)
#sss.voltage = voltage_chan.data
#sss._align_spikes()

    
plt.figure(figsize=(15, 5))
plt.plot(sss.voltage, alpha = 0.5, color = 'r')
plt.eventplot(sss.spike_indices, alpha = 0.5, linelengths=np.max(sss.voltage)*2, colors='b')
plt.show()

#%%
ss_sorted_path = '../data/spike_sorted_temp/'
for fn in f_names:
    ss_sorted_fn = find_file(os.path.split(fn)[1] + '.pkl', ss_sorted_path)
    with open(ss_sorted_fn, 'rb') as input:
        sss = pickle.load(input)
        spike_indices = sss.spike_indices
    with open(os.path.join('../data/spike_sorted_temp/', os.path.basename(fn) + '_ss.pkl'), 'wb') as output:
        pickle.dump(spike_indices, output, pickle.HIGHEST_PROTOCOL)
        
    
    






    
    
    
    
    