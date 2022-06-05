import numpy as np
import scipy
import librosa

def stat_features(array):
    mean= np.mean(array)
    median = np.median(array)
    std= np.std(array)
    kurtosis = scipy.stats.kurtosis(array, axis=0, bias=True)
    skewness = scipy.stats.skew(array, axis=0, bias=True)

    return (mean,median,std,kurtosis,skewness)


def mfcc(array,sr=4000):
    mfccs = librosa.feature.mfcc(array, sr=sr)
    print(mfccs.shape)

    return mfccs


# array = np.arange(0,100,50)
# print(stat_features(array))
# print(mfcc(array))


