"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import numpy as np
import math
import toolbox.hdf5

def generate_voltage_trace(simple_spike_templates, complex_spike_templates, ss_frequency=100, cs_frequency=0.5, time=60, dt=1/100e3, noise_std=1000, cs_scaling=1.0):
    """Generate voltage traces"""

    # Load the waveforms
    ss = []
    if isinstance(simple_spike_templates, list) or isinstance(simple_spike_templates, tuple):
        for i in range(len(simple_spike_templates)):
            ss.append(toolbox.hdf5.hdf5_load(simple_spike_templates[i])['data'])
    else:
        ss = [simple_spike_templates]

    cs = []
    if isinstance(complex_spike_templates, list) or isinstance(complex_spike_templates, tuple):
        for i in range(len(complex_spike_templates)):
            cs.append(toolbox.hdf5.hdf5_load(complex_spike_templates[i])['data'])
    else:
        cs = [complex_spike_templates]

    # Generate the voltage timeseries
    voltage = np.zeros(math.ceil(time/dt))

    # Pick times for each of the simple spikes
    num_spikes = math.ceil(ss_frequency * time)

    # Choose random locations for each of the simple spikes
    for i in range(0, len(ss)):
        indices = np.sort(np.random.randint(0, len(voltage) - len(ss[i]), num_spikes))
        # Ensure that no indices are within 2ms of each other
        refractory_violations = np.where(np.diff(indices) < math.ceil(2e-3 / dt))[0]
        if len(refractory_violations) > 0:
            indices[refractory_violations] + math.ceil(2e-3 / dt)

        # Place the spikes
        for j in range(0, len(indices)):
            voltage[indices[j]:indices[j]+len(ss[i])] += ss[i]

    # Place the complex spikes
    num_cs_spikes = math.ceil(cs_frequency * time)
    for i in range(0, len(cs)):
        indices = np.sort(np.random.randint(0, len(voltage) - len(cs[i]), num_cs_spikes))
        print('Complex spike indices: ' + str(indices))
        # Ensure that no indices are within 2ms of each other
        refractory_violations = np.where(np.diff(indices) < math.ceil(2e-3 / dt))[0]
        if len(refractory_violations) > 0:
            indices[refractory_violations] + math.ceil(2e-3 / dt)

        # Place the spikes
        for j in range(0, len(indices)):
            voltage[indices[j]:indices[j] + len(cs[i])] += np.array(cs[i]) * cs_scaling

    # Add noise
    voltage = voltage + np.random.randn(len(voltage)) * noise_std

    return voltage

def generate_bursting_voltage_trace(simple_spike_templates, complex_spike_templates, ss_burst_length=25, ss_frequency=100, cs_frequency=0.5, time=60, dt=1/100e3, noise_std=1000, cs_scaling=1.0):
    """Generate voltage traces"""

    # Load the waveforms
    ss = []
    if isinstance(simple_spike_templates, list) or isinstance(simple_spike_templates, tuple):
        for i in range(len(simple_spike_templates)):
            ss.append(toolbox.hdf5.hdf5_load(simple_spike_templates[i])['data'])
    else:
        ss = [simple_spike_templates]

    cs = []
    if isinstance(complex_spike_templates, list) or isinstance(complex_spike_templates, tuple):
        for i in range(len(complex_spike_templates)):
            cs.append(toolbox.hdf5.hdf5_load(complex_spike_templates[i])['data'])
    else:
        cs = [complex_spike_templates]

    # Generate the voltage timeseries
    voltage = np.zeros(math.ceil(time/dt))

    # Pick times for each of the simple spikes
    num_spikes = int(math.ceil(ss_frequency * time))

    # Choose a random location for a burst
    bursts = int(math.ceil(num_spikes / ss_burst_length))

    for i in range(0, len(ss)):
        start_index = np.random.randint(0, int(len(voltage) - ss_burst_length * (2e-3/dt)), bursts)
        for j in range(len(start_index)):
            for k in range(ss_burst_length):
                if start_index[j] + len(ss[i]) > len(voltage):
                    continue
                voltage[start_index[j]:start_index[j] + len(ss[i])] += ss[i]
                start_index[j] += np.random.randint(int(round(1e-3/dt)), int(round(5e-3/dt)))

    # Choose random locations for each of the simple spikes
    for i in range(0, len(ss)):
        indices = np.sort(np.random.randint(0, len(voltage) - len(ss[i]), num_spikes))
        # Ensure that no indices are within 2ms of each other
        refractory_violations = np.where(np.diff(indices) < math.ceil(2e-3 / dt))[0]
        if len(refractory_violations) > 0:
            indices[refractory_violations] + math.ceil(2e-3 / dt)

        # Place the spikes
        for j in range(0, len(indices)):
            voltage[indices[j]:indices[j] + len(ss[i])] += ss[i]

    # Place the complex spikes
    num_cs_spikes = math.ceil(cs_frequency * time)
    for i in range(0, len(cs)):
        indices = np.sort(np.random.randint(0, len(voltage) - len(cs[i]), num_cs_spikes))
        print('Complex spike indices: ' + str(indices))
        # Ensure that no indices are within 2ms of each other
        refractory_violations = np.where(np.diff(indices) < math.ceil(2e-3 / dt))[0]
        if len(refractory_violations) > 0:
            indices[refractory_violations] + math.ceil(2e-3 / dt)

        # Place the spikes
        for j in range(0, len(indices)):
            voltage[indices[j]:indices[j] + len(cs[i])] += np.array(cs[i]) * cs_scaling

    # Add noise
    voltage = voltage + np.random.randn(len(voltage)) * noise_std

    return voltage


