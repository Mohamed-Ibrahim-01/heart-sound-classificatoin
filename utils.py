import numba
import os
import numpy as np
import zipfile
import shutil
import glob
import re
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
    name, zip_path = dataset_info['NAME'], dataset_info["ZIP_PATH"]
    dataset_dir = f"{Dirs['DATASETS']}{os.sep}{name}"
    if(os.path.exists(dataset_dir)):
        shutil.rmtree(dataset_dir)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    return dataset_dir

def _folder_label(folder_path):
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

def _load_pascal_array(pascal_df):
    sounds = []
    pascal_df.apply(lambda row : add_sound(row, sounds), axis=1, raw=True)
    return np.array(sounds, dtype=object)

# return sounds np 2D array (with labels) [[array1, label][array2, label]]
def load_pascal():
    pascal_dir = _load_dataset_dir(PascalInfo)
    pascal_df = _load_pascal_df(pascal_dir)
    pascal_array = _load_pascal_array(pascal_df)

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
    with open(f'training-{dataset_char}\RECORDS-{label}') as f:
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
    datafraame = build_physio_dataframe()
    sounds  = _load_pascal_array(datafraame)
    return sounds

def db6_wavelet_denoise(x):
    a5, d5, d4, d3, d2, d1 = pywt.wavedec(x, 'db6', level=5)
    reconstructed = pywt.waverec([a5, d5, np.zeros_like(d4), d3, d2, np.zeros_like(d1)], 'db6')
    return reconstructed
