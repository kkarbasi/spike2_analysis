from scipy.signal import resample
import numpy as np

def resample_to_freq(data, source_frequency, target_frequency):
    """
    resamples the input data from source_frequency to target_frequency
    """
    resample_ratio = float(target_frequency) / source_frequency
    if resample_ratio == 1:
        return data
    else:
        target_n_samples = int(np.size(data) * resample_ratio)
        resampled_data = resample(data, target_n_samples)
        return resampled_data
