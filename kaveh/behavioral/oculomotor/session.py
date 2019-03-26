import numpy as np
from scipy.signal import savgol_filter
from kaveh.toolbox import find_first
from scipy.signal import find_peaks



class session:

    def __init__(self,
            HT, t_HT,
            VT, t_VT,
            HE, t_HE,
            VE, t_VE,
            eye_fs, eye_dt):

        self.HT = HT
        self.t_HT = t_HT
        self.VT = VT
        self.t_VT = t_VT
        self.HE = HE
        self.t_HE = t_HE
        self.VE = VE
        self.t_VE = t_VE
        self.fs = eye_fs
        self.dt = eye_dt


    def _calc_target_velocity(self):
        VT_v_filtered = savgol_filter(np.squeeze(self.VT), window_length=15, polyorder=2, deriv=1, delta = self.dt)
        HT_v_filtered = savgol_filter(np.squeeze(self.HT), window_length=15, polyorder=2, deriv=1, delta = self.dt)
        self.T_v_filtered = np.linalg.norm(np.vstack((VT_v_filtered, HT_v_filtered)), axis = 0)

    def _calc_saccade_velocity(self):
        VE_v_filtered = savgol_filter(np.squeeze(self.VE), window_length=15, polyorder=2, deriv=1, delta = self.dt)
        HE_v_filtered = savgol_filter(np.squeeze(self.HE), window_length=15, polyorder=2, deriv=1, delta = self.dt)
        self.E_v_filtered = np.linalg.norm(np.vstack((VE_v_filtered, HE_v_filtered)), axis = 0)


    def _detect_target_jumps(self, v_thresh = 200, onoff_thresh = 20):
        '''
        Detecting target jumps based on their velocity

        :param v_thresh: velocity threshold for detecting targets (deg/s)
        :param onoff_thresh: velocity threshold for detecting target onset and offset (deg/s)
        '''
        rising = self.T_v_filtered > v_thresh
        rising[1:][rising[1:] & rising[:-1]] = False

        # remove detected saccades that are withing 0.005 s of another detected saccade
        target_times = self.t_VT[rising]
        target_times = np.delete(target_times, np.where(np.diff(target_times)<0.005))
        rising = np.isin(self.t_VT, target_times)

        # Here we find the onset and offset of the detected targets (whose velocity cross v_thresh).
        # We require that the target velocity remain under the threshold
        # for 5 ms before and after the target (minimum inter-target interval)
        pattern = [True for i in range(int(np.ceil(0.005 / self.dt)))]
        target_indices = np.where(rising)[0]

        below_onset_offset_thresh = self.T_v_filtered < onoff_thresh
        self.target_onsets = []
        self.target_offsets = []
        for si in target_indices:
            self.target_onsets.append(find_first(below_onset_offset_thresh, start = si, direction = 'backward', pattern = pattern))
            self.target_offsets.append(find_first(below_onset_offset_thresh, start = si, direction = 'forward', pattern = pattern))

        self.target_onsets = np.unique(np.array(self.target_onsets))
        self.target_offsets = np.unique(np.array(self.target_offsets))

        # Here we delete the targets that have more than 1 prominent peaks ( with peak height = 20)
        to_delete = []
        for i, (son, soff) in enumerate(zip(self.target_onsets, self.target_offsets)):
            peaks = find_peaks(self.T_v_filtered[son:soff+1], prominence=1)[0]
            if (np.size(peaks) > 1):
                to_delete.append(i)
            
            
        self.target_onsets = np.delete(self.target_onsets, to_delete)
        self.target_offsets = np.delete(self.target_offsets, to_delete)

        # Here we delete the targets with amplitude less than 0.5 degrees
        H_target_amp = self.HT[self.target_offsets] - self.HT[self.target_onsets]
        V_target_amp = self.VT[self.target_offsets] - self.VT[self.target_onsets]
        target_amp = np.linalg.norm(np.hstack((H_target_amp, V_target_amp)), axis = 1)
        to_delete = []

        for i, (son, soff) in enumerate(zip(self.target_onsets, self.target_offsets)):
            if target_amp[i] < 0.5:
                to_delete.append(i)
        self.target_onsets = np.delete(self.target_onsets, to_delete)
        self.target_offsets = np.delete(self.target_offsets, to_delete) 



    def _detect_saccades(self, v_thresh = 80, onoff_thresh = 20):
        '''
        Detecting saccade jumps based on their velocity

        :param v_thresh: velocity threshold for detecting saccades (deg/s)
        :param onoff_thresh: velocity threshold for detecting saccade onset and offset (deg/s)
        '''
        rising = self.E_v_filtered > v_thresh
        rising[1:][rising[1:] & rising[:-1]] = False

        # remove detected saccades that are withing 0.010 s of another detected saccade
        saccade_times = self.t_VE[rising]
        saccade_times = np.delete(saccade_times, np.where(np.diff(saccade_times)<0.010))
        rising = np.isin(self.t_VE, saccade_times)

        # Here we find the onset and offset of the detected saccades (whose velocity cross v_thresh).
        # We require that the saccade velocity remain under the threshold
        # for 5 ms before and after the saccade (minimum inter-saccade interval)
        pattern = [True for i in range(int(np.ceil(0.010 / self.dt)))]
        saccade_indices = np.where(rising)[0]

        below_onset_offset_thresh = self.E_v_filtered < onoff_thresh
        self.saccade_onsets = []
        self.saccade_offsets = []
        for si in saccade_indices:
            self.saccade_onsets.append(find_first(below_onset_offset_thresh, start = si, direction = 'backward', pattern = pattern))
            self.saccade_offsets.append(find_first(below_onset_offset_thresh, start = si, direction = 'forward', pattern = pattern))

        self.saccade_onsets = np.unique(np.array(self.saccade_onsets))
        self.saccade_offsets = np.unique(np.array(self.saccade_offsets))
        # Here we delete the saccades that have more than 1 prominent peaks ( with peak height = 20)
        to_delete = []
        for i, (son, soff) in enumerate(zip(self.saccade_onsets, self.saccade_offsets)):
            peaks = find_peaks(self.E_v_filtered[son:soff], prominence=15)[0]
            if (np.size(peaks) > 1):
                to_delete.append(i)
            peaks = find_peaks(self.E_v_filtered[son:soff], prominence=1)[0]
            if (np.size(peaks) > 3):
                to_delete.append(i)            
        self.saccade_onsets = np.delete(self.saccade_onsets, to_delete)
        self.saccade_offsets = np.delete(self.saccade_offsets, to_delete)

        # Here we delete the saccades with amplitude less than 0.5 degrees
        H_saccade_amp = self.HE[self.saccade_offsets] - self.HE[self.saccade_onsets]
        V_saccade_amp = self.VE[self.saccade_offsets] - self.VE[self.saccade_onsets]
        saccade_amp = np.linalg.norm(np.hstack((H_saccade_amp, V_saccade_amp)), axis = 1)
        to_delete = []

        for i, (son, soff) in enumerate(zip(self.saccade_onsets, self.saccade_offsets)):
            if saccade_amp[i] < 0.5:
                to_delete.append(i)
        self.saccade_onsets = np.delete(self.saccade_onsets, to_delete)
        self.saccade_offsets = np.delete(self.saccade_offsets, to_delete) 

    @property
    def saccade_onset_times(self):
        return self.t_VE[self.saccade_onsets]

    @property
    def saccade_offset_times(self):
        return self.t_VE[self.saccade_offsets]

    @property
    def target_onset_times(self):
        return self.t_VT[self.target_onsets]

    @property
    def target_offset_times(self):
        return self.t_VT[self.target_offsets]



