import numba
import os
import numpy as np
import zipfile
import shutil
import glob
import re
import math
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pywt
BASE_DIR = os.getcwd()

Dirs = {
    "BASE" : BASE_DIR,
    "DATASETS" : f"{BASE_DIR}{os.sep}datasets",
}

PascalInfo = {
    "NAME" : "pascal",
    "ZIP_NAME" : "Pascal.zip",
    "DRIVE_ID" : "1nkSqb_B-uqv7-zka9HlmAcLkoDET2mqk",
    "ZIP_PATH" : ""
}

PascalInfo["ZIP_PATH"] = f"{BASE_DIR}{os.sep}{PascalInfo['ZIP_NAME']}"

def _load_dataset_dir(dataset_info):
    """
    extracts data files into dataset directory

    Parameters
    ---------------
    dataset_info: dictionary
    dictionary including the dataset's metadata

    Returns
    ---------------
    dataset_dir: string
    dataset directory path
    
    """
    name, zip_path = dataset_info['NAME'], dataset_info["ZIP_PATH"]
    dataset_dir = f"{Dirs['DATASETS']}{os.sep}{name}"
    if(os.path.exists(dataset_dir)):
        shutil.rmtree(dataset_dir)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    return dataset_dir

def _folder_label(folder_path):
    """
    gets the label of the signals in a folder

    Parameters
    ---------------
    folder_path: string
    path of the directory containing the dataset's signals

    Returns
    ---------------
    label: string
    label of signals in a folder
    
    """
    label = re.findall(r"murmur|normal|extrastole", folder_path.lower())
    if len(label) == 0:
        return "other"
    return label[0] if label[0] != "extrastole" else "extrasystole"

def _pascal_folders(pascal_dir):
    
    zip_files = glob.glob(f'{pascal_dir}{os.sep}*.zip')
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pascal_dir)
    folders = glob.glob(f'{pascal_dir}{os.sep}*{os.sep}')
    folders.pop()
    return folders

def _load_pascal_df(pascal_dir):
    sounds = []
    folders = _pascal_folders(pascal_dir)
    for folder_path in folders:
        label = _folder_label(folder_path)
        if label != "other":
            wav_files = glob.glob(f'{folder_path}{os.sep}*.wav')
            for sound_path in wav_files:
                sound_name = os.path.basename(sound_path).split(".")[0]
                sounds.append([sound_name, sound_path, label])
    return pd.DataFrame(sounds, columns=["name", "path", "label"])


@numba.jit(forceobj=True)
def add_sound(curr_row, sounds):
    _, path, label = tuple(curr_row)
    _, sound = wavfile.read(path)
    sound = sound.astype(np.float32)
    sounds.append([sound, label])
    return curr_row

def _load_dataset_array(dataset_df):
    sounds = []
    dataset_df.apply(lambda row : add_sound(row, sounds), axis=1, raw=True)
    return np.array(sounds, dtype=object)

# return sounds np 2D array (with labels) [[array1, label][array2, label]]
def load_pascal():
    pascal_dir = _load_dataset_dir(PascalInfo)
    pascal_df = _load_pascal_df(pascal_dir)
    pascal_array = _load_dataset_array(pascal_df)

    return pascal_array, pascal_df

def plot_hs(path, x=[], dur=1.5, verts=[], sr=4000):
    y, sr = librosa.load(path, sr=sr, duration=dur) if len(x) == 0 else (x, sr)
    _, ax = plt.subplots(nrows=1, figsize=(20,4))
    for line in verts:
        ax.axvline(x=line, color="red", ls='--')
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=f'{path.split("/")[-1]}')
    plt.show()
    return (y, sr)


def normalize(signal, mode="min-max"):
    base = min(signal) if mode == "min-max" else 0
    range = max(signal) - base
    normalized = [(x-base)/range for x in signal]
    return np.array(normalized)


def load_physio_dataset(dataset_char,label):
    with open(f'training-{dataset_char}{os.sep}RECORDS-{label}') as f:
        lines = f.readlines()
    records = []
    for index in range(len(lines)) :
        lines[index] = lines[index].strip()
        # print(lines[index])
        records.append([lines[index],f'training-{dataset_char}{os.sep}{lines[index]}.wav',label])
    return records 


def build_physio_dataframe():
    datasets = ['a','b','c','d','e','f']
    records = []
    for char in datasets:
        records.extend(load_physio_dataset(char,'normal'))
        records.extend(load_physio_dataset(char,'abnormal'))
    return  pd.DataFrame(records,columns=["name", "path", "label"])


def load_physioNet():
    """
    builds physioNet dataframe with features and labels
    Returns
    -------------
    sounds: 2D array of floats
    2D array containing the sound files numeric data
    """
    dataframe = build_physio_dataframe()
    sounds  = _load_dataset_array(dataframe)
    return sounds


def db6_wavelet_denoise(x):
    a5, d5, d4, d3, d2, d1 = pywt.wavedec(x, 'db6', level=5)
    reconstructed = pywt.waverec([a5, d5, np.zeros_like(d4), d3, d2, np.zeros_like(d1)], 'db6')
    return reconstructed

def features_histo(dfs):
    """
    draws histograms of features

    Parameters
    -------------
    dfs: array of dataframes
    array containing all the dataframes to plot

    """
    features_names = dfs[0].columns
    fig, axs = plt.subplots(nrows=9, ncols=3, figsize=(20,30),
                            gridspec_kw={'hspace': 0.3})
    for df in dfs:
        for i, feature in enumerate(features_names):
            ax = axs[math.floor(i/3)][i%3]
            ax.hist(df[feature], bins=20)
            ax.set(title=feature)
    plt.show()


def load_heartsound_features():
    """
    takes the physioNet dataset and outputs the balanced classes dataset, the normal patients' dataset,
    and the abnormal patients' dataset. This is done for easier usage and analysis of the data

    Returns
    -----------
    hs_df: pandas dataframe
    dataframe with the balanced classes

    normal_df: pandas dataframe
    dataframe of normal patients' records

    abnormal_df: pandas dataframe
    dataframe of abnormal patients' records

    """
    hs_df = pd.read_csv("physioNet.csv")
    hs_df = hs_df.drop(columns=["Unnamed: 0"])
    hs_df = hs_df.sort_values(by='Label', ascending=False)
    hs_df = hs_df.tail(hs_df.shape[0]-4000)

    normal_df = hs_df[hs_df["Label"] == "normal"]
    abnormal_df = hs_df[hs_df["Label"] == "abnormal"]
    return hs_df, normal_df, abnormal_df
