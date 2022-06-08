import math
import librosa.util
import numpy as np


def segment_sound(record,label, n_cycles=5, samplerate=4000):
    segement_duration = n_cycles*0.8
    record_duration = len(record)/samplerate #the dataset is sampled at 4K
    segments_num = math.floor(record_duration/segement_duration)
    segment_pts = math.floor(segement_duration*samplerate)
    segs_arr = []
    single_seg = []
    for i in range(segments_num):
        single_seg = record[i*segment_pts : segment_pts*(i+1)]
        segs_arr.append([single_seg,label])
    return segs_arr


def build_segements(data_arr):
    segments = []
    for record in data_arr:
        segments.extend(segment_sound(record[0], record[1]))
    return segments


def frame_hs(x, sr, frame_time, overlap_time):
    if(frame_time < overlap_time):
        raise ValueError("overlap cannot be larger than frame")
    frame_length = int(frame_time*sr)
    overlap_length = int(overlap_time*sr)
    hop_length = frame_length - overlap_length
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length, axis=0)
    return frames


def shannon_energy(x):
    x_sq = x**2
    x_sq = x_sq[x_sq != 0]
    N = len(x)
    return -1/N * np.sum(x_sq * np.log(x_sq))


def framed_shannon_envelope(x, sr):
    envelope = []
    frames = frame_hs(x, sr, 0.002, 0.001)
    for frame in frames:
        se = shannon_energy(frame)
        if not math.isnan(se) and not math.isinf(se):
            envelope.append(se)
    envelope_sr = int(sr*len(envelope)//len(x))
    return np.array(envelope), envelope_sr

def shannon_envelope(x):
    x_sq = x**2
    x_sq = x_sq[x_sq != 0]
    return -1*(x_sq)*(np.log(x_sq))


def threshold_se(se, percent=60):
    max = np.max(se)
    print(max)
    th_envelope = np.copy(se)
    th_envelope[th_envelope < (percent/100 * max)] = 0
    return th_envelope

def s1_peaks_se(se, s1_dist, merging_dist):
    pass

"""
def murmur_elimination(input_signal):
    denoised_signal = denoising(input_signal)
    normalized_signal = normalization(denoised_signal)
    envelope = automatic_low_pass_filter(normalized_signal)
    normalized_envelope = normalization(envelope)
    cut_off_frequency = get_cut_off_frequency(normalized_envelope)
    for i in range(len(normalized_envelope)):
        if(normalized_envelope[i] < cut_off_frequency):
            filtered_normalized_envelope = np.delete(normalized_envelope, i)
    
def get_cut_off_frequency(Normalized_envelope): #get the cut off freq required to filter murmur (threshold in freq domain below which the elimination occur)
    valley_pts_indices = get_valley_pts_indices(Normalized_envelope)
    for i in range(len(valley_pts_indices)):
        if (Normalized_envelope[i] < 0.2): # i + j -> complex then 0.2?
            threshold = Normalized_envelope[i] # i + j ?
        else:
            threshold = 200
    return threshold #the cut off freq for filtering

from scipy.signal import argrelmin
def get_valley_pts_indices(Normalized_envelope):
    valley_pts_indices = argrelmin(Normalized_envelope)
    return valley_pts_indices #return a list containing the indices of the valley pts

import librosa
def fft_evelope(sound_fft, lf=5):
    pad_left = np.append(np.zeros(lf), sound_fft, 0)
    padded_fft = np.append(pad_left, np.zeros(lf), 0)
    frames = librosa.util.frame(padded_fft, frame_length=2*lf+1, hop_length=1, axis=0)
    print(frames.shape)
    efft = np.average(frames, axis=1)
    return efft

from scipy.fft import fft
def automatic_low_pass_filter(normalized_signal):
    ffth = fft(normalized_signal)z
    efft = fft_evelope(ffth)
    get_valley_pts_indices(efft)
"""
