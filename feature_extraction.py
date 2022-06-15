import numpy as np
import scipy.stats
import math
import librosa
import pandas as pd
import utils
import segmentation

def get_max_amplitude(signal):
    return max(list(signal))

def get_dominant_frequency(signal,sampling_rate =4000):
    """
    Calculates the dominant frequency in a signal using fourier transfrom 

    Parameters
    -----------------

    signal: array 
    array of signal values to be evaluated

    sampling_rate: int
    sampling rate of the audio signal

    Returns
    -----------------

    positive_frequencies[peak_frequency]: float
    value of the dominant frequency in the signal
    """
    fourier = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(np.array(signal).size, d=1.0/sampling_rate) 
    positive_frequencies = frequencies[np.where(frequencies >= 0)] 
    magnitudes = abs(fourier[np.where(frequencies >= 0)])
    peak_frequency = np.argmax(magnitudes)
    return positive_frequencies[peak_frequency]


def get_entropy(signal):
    """
    calculates the entropy of a signal

    Parameters
    -----------------

    signal: array 
    array of signal values to be evaluated

    Returns
    -----------------

    entropy: float
    value of the entropy of the signal
    """
    number_of_bins = math.ceil(math.sqrt(len(signal))) 
    step = 1.0/number_of_bins
    counts,_  = np.histogram(signal, bins=np.arange(0,max(signal)+step,step))
    return scipy.stats.entropy(counts)


def stat_features(array):
    """
    Calculates the statistical features of the signal [mean,median,std,kurtosis,skewness,iqr,
    first percentile,second percentile,third percentile]

    Parameters
    -----------------

    array: array 
    array of signal values to be evaluated


    Returns
    -----------------

   mean: float 
   signal mean value
   
   median: float 
   signal median value

   std: float 
   standard deviation value of the signal

   kurtosis: float 
   signal kurtosis value

   skewness: float 
   signal skewness value

   iqr: float 
   signal interquartile range value

   first_percentile: float
   signal first percentile value 

   second_percentile: float 
   signal second percentile value

   third_percentile: float 
   signal third percentile value
    """
    mean= np.mean(array)
    median = np.median(array)
    std= np.std(array)
    kurtosis = scipy.stats.kurtosis(array, axis=0, bias=True)
    skewness = scipy.stats.skew(array, axis=0, bias=True)
    iqr = scipy.stats.iqr(array)
    first_percentile = np.percentile(array,25)
    second_percentile = np.percentile(array,50)
    third_percentile = np.percentile(array,75)
    return mean,median,std,kurtosis,skewness,iqr,first_percentile,second_percentile,third_percentile

def mfcc(array,sampling_rate=4000):
    """
    Calculates the Mel-Frequency Cepstral Coefficients (MFCCs) of the signal

    Parameters
    -----------------

    array: array 
    array of signal values to be evaluated

    sampling_rate:
    sampling rate of the audio signal


    Returns
    -----------------
    mfccs: list 
    flattened list of means of coefficients 
    """
    mfccs = librosa.feature.mfcc(y=array, sr=sampling_rate,n_mfcc=13)
    mfccs = np.mean(mfccs,axis=1)
    return list(mfccs.flatten())

# function to build the whole data frame of features with labels  
# params 2d array of segments with labels 

def build_features_df(segments):
    """
    constructs a dataframe of the signal segments where each row corresponds to a segment,
    and columns are the segments features and signal label

    Parameters
    -----------------

    segments: array 
    array of segments values to use

    Returns
    -----------------
    dataframe: 2D list of floats and strings
    segments features and labels
    """
    data , labels = separate_labels(segments=segments)
    # print(labels)
    dataframe = []
    for index ,element in enumerate(data) :
        features = list(extract_segment_features(element))
        features.append(labels[index])
        dataframe.append(features)
    # convert the data frame multidimensional array to pandas data frame
    return dataframe
    

def separate_labels(segments):
    """
    separates segments labels from segments data values

    Parameters
    -----------------

    segments: array 
    array of segments with their labels

    Returns
    -----------------
    data: 2D list of floats
    segments data values to use in computations

    labels: list of strings
    segments labels 

    """
    data =[]
    labels = []
    for row in segments:
        data.append(row[0])
        labels.append(row[1])
    return data,labels


#amplitude | freq | entropy |mean | median | std | kurtosis 
# | skewness | iqr | first_percentile | second_percentile | third_percentile | mfccs 
def extract_segment_features(segment):
    """
    calculates signal feature values and combines them into an array

    Parameters
    -----------------

    segment: array 
    array of segment values to use

    Returns
    -----------------
    features: list 
    feature values
    """
    features =[]
    features.extend([get_max_amplitude(list(segment)),get_dominant_frequency(segment),get_entropy(list(segment))])
    features.extend(list(stat_features(np.array(segment))))
    features.extend(mfcc(np.array(segment)))
    return features


def construct_dataframe(dataset_name):
    """
    constructs a dataframe of the signal segments where each row corresponds to a segment,
    and columns are the segments features and signal label

    Parameters
    -----------------

    dataset_name: string
    name of the dataframe 

    Returns
    -----------------
    dataframe: pandas dataframe
    dataframe containing segments features and labels
    """
    segments = []
    if dataset_name == "pascal":
        records ,df= utils.load_pascal()
        segments = segmentation.build_segements(records, sr=4000)
    else:
        records = utils.load_physioNet()
        segments = segmentation.build_segements(records, sr=2000)
    features_matrix = build_features_df(segments)
    
    dataframe = pd.DataFrame(features_matrix,columns=["Max_Amplitude" , "Dominant_Freq" , "Entropy", "Mean" ,"Median" ,"STD", "Kurtosis" 
    ,"Skewness" ,"IQR", "First_Percentile", 
    "Second_Percentile", "Third_Percentile","MFCC1",
    "MFCC2","MFCC3","MFCC4","MFCC5",
    "MFCC6","MFCC7","MFCC8","MFCC9","MFCC10"
    ,"MFCC11","MFCC12","MFCC13","Label"])
    dataframe.to_csv(f"{dataset_name}.csv")
    return dataframe
