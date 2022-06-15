from sklearn.model_selection import GridSearchCV
from IPython.display import display_html, display
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from utils import load_physioNet, db6_wavelet_denoise, load_heartsound_features
import IPython.display as ipd
import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


physio_arr = load_physioNet()
normal_physio = physio_arr[physio_arr[:,1] == 'normal']
abnormal_physio = physio_arr[physio_arr[:,1] == 'abnormal']
hs_df, normal_df, abnormal_df = load_heartsound_features()

def _get_random_samples(n_samples, seed):
    """
    function to pick random samples from the dataset

    Parameters
    ----------

    n_sample: int
        the no of samples to be picked randomly

    seed: int
        the seed value passed to fn np.random.seed to keep 
        the same value of seed constant so at each time the 
        cell is run the output random records are the same

    Returns
    -------

    sample_sounds: array
        array of random sample records

    """
    if(n_samples > 10 or seed > 500):
        raise ValueError("Max samples is 10 and is even, seed max is 500")

    np.random.seed(seed)
    a = np.random.randint(0, 500)

    normal_samples = [(normal_physio[a+i][0], f"Noraml Sample {i}") for i in range(n_samples)]
    abnormal_samples = [(abnormal_physio[a+i][0], f"Abnoraml Sample {i}") for i in range(n_samples)]
    sample_sounds = normal_samples + abnormal_samples
    return sample_sounds

def view_downsampled_dist():
    """
    function for data distriburion visualization 
    plots a pie chart showing the contribution of each class of the segments extracted from the dataset

    """
    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
    normal_cnt, abnormal_cnt = tuple(hs_df["Label"].value_counts(normalize=True))
    axs[1].pie([normal_cnt,abnormal_cnt], labels=["Normal", "Abnormal"], autopct='%.0f%%', textprops={'fontsize':15})
    axs[0].axis('off')
    axs[2].axis('off')
    plt.show()

def view_random_samples(n_samples=2, seed=0):
    """
    function to play the random samples of the records from the dataset

    Parameters
    ----------
    n_samples: int
        no of random samples

    seed: int
        the seed value passed to fn np.random.seed to keep 
        the same value of seed constant so at each time the 
        cell is run the output random records are the same

    """
    sample_sounds = _get_random_samples(n_samples, seed)
    for sound in sample_sounds:
        sample_sound, title = sound
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
        librosa.display.waveshow(sample_sound, sr=2000, ax=axs[0])
        sound_stft = librosa.stft(sample_sound)
        sound_spectrogram = librosa.amplitude_to_db(abs(sound_stft))
        librosa.display.specshow(sound_spectrogram, sr=2000, x_axis='time', y_axis='log', ax=axs[1])
        axs[0].set_title(title)

        plt.show()
        display(ipd.Audio(sample_sound, rate=2000))


def view_files_dist():
    """
    function for data distriburion visualization 
    plots a pie chart showing the contribution of each class in the records of in the dataset

    """
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    classes = pd.DataFrame(physio_arr[:,1])
    normal_cnt, abnormal_cnt = classes.value_counts(normalize=True)
    ax.pie([normal_cnt,abnormal_cnt]
           , labels=["Normal", "Abnormal"]
           , autopct='%.0f%%', textprops={'fontsize':15}
           )
    ax.set_title("Normal & Abnormal Sound Files")
    plt.show()

def view_seconds_dist():
    """
    function to visualize th distribution of the total duration of each class in seconds
    pie chart plot the total duration contribution of each class 
    """
    _, axs = plt.subplots(ncols=2, figsize=(20,4))
    abnormal_seconds = [len(sound)/2000 for sound in abnormal_physio[:, 0]]
    normal_seconds = [len(sound)/2000 for sound in normal_physio[:, 0]]

    axs[0].hist(normal_seconds, bins=25, alpha = 0.7)
    axs[0].hist(abnormal_seconds, bins=25, alpha = 0.7)
    axs[0].set_xticks(range(0, 130, 10))
    axs[0].set_title("Normal & Abnormal Seconds Distribution")
    axs[0].legend(["Normal", "Abnormal"])

    normal_length = np.sum(normal_seconds)
    abnormal_length = np.sum(abnormal_seconds)
    total_seconds = normal_length+abnormal_length

    axs[1].pie([normal_length, abnormal_length]
               , labels=["Normal", "Abnormal"]
               , autopct='%.0f%%', textprops={'fontsize':15}
               )
    axs[1].set_title("Total Normal & Abnormal Seconds Distribution")
    plt.show()

def view_wavlet_denoising():
    """
    function to plot a random abnormal heart sound sample & the same sample after denoising 
    """
    abnormal_sound = _get_random_samples(1, 0)[1][0]
    sample_sounds = [abnormal_sound, db6_wavelet_denoise(abnormal_sound)]
    labels = ["Abormal", "Abnormal Denoised"]
    for i, sound in enumerate(sample_sounds):
        sample_sound, title = sound, labels[i]
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,3))
        librosa.display.waveshow(sample_sound[:2000], sr=2000, ax=axs[0])
        X = librosa.stft(sample_sound)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=2000, x_axis='time', y_axis='log', ax=axs[1])
        axs[0].set_title(title)
        plt.show()

def compare(classifiers, X, y, grid_search=[]):
    """
    function to compare the classifiers models with respect to each other

    parameters
    ----------
    classifiers: list
        list of the classifiers to be used on the data
    
    X: array
        the inputs of the dataset
    
    y: array
        the labels of the dataset
    
    grid_search: list

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0,shuffle=True)
    data = ((X_train, y_train), (X_test, y_test))
    train, test = data
    fitted_clfs, train_reports, test_reports = [], [], []
    best_params = dict()
     
    for clf in classifiers: 
        fitted_clfs.append(fit_data(clf, train, grid_search))

    for clf in fitted_clfs: 
        train_reports.append(clf_report(clf, train))
        test_reports.append(clf_report(clf, test))

    log_reports(zip(train_reports, test_reports))
    confusion_compare(fitted_clfs, test)
    roc_compare(fitted_clfs, test)
     
    for clf in fitted_clfs:
        best_params[clf[0]] = clf[1].get_params(deep=False)

    return fitted_clfs, best_params


def fit_data(clf, data, grid_search):
    """
    Helper function for compare function

    Parameters
    ----------
    clf: string
        name of the classifier to be used on the data 
    """
    inputs, targets = data
    classifier = clf['cached']
    if grid_search:
        if clf['name'] in set(grid_search):
            print(f"classirier name {clf['name']}")
            classifier = GridSearchCV(clf['method'](), clf['parameters'], cv=5, scoring ="f1_macro")
    classifier.fit(inputs, targets);
    return (clf['name'], classifier)


def clf_report(clf, data):
    """
    helper function for classification_report function
    """
    clf_name, classifier = clf
    inputs, lables = data
    predictions = classifier.predict(inputs)
    report = classification_report(lables, predictions, output_dict=True)
    return (clf_name, report)


def log_reports(reports):
    """
    helper function for classification_report function
    """
    for report in reports:
        (clf_name, train_report),(_,test_report) = report
        display_html(f'<h3>{clf_name}</h3>', raw=True)
        train_report_df = pd.DataFrame.from_dict(train_report)
        test_report_df = pd.DataFrame.from_dict(test_report)
        dfs = {'Train': train_report_df, 'Test': test_report_df}
        display_dfs(dfs)

def display_dfs(dfs, gap=20, justify='center'):
    """
    Displaying dataframe horizontally to be visually good

    Parameters
    ----------
    dfs : dict
        The dataframes to display with titles

    gap : int
        The horizontal gap sapce between 2 dataframes
    """

    html = ""
    for title, df in dfs.items():  
        df_html = df.head(n=11)._repr_html_()
        cur_html = f'<div style="text-align:center;"> <h5>{title}</h5> {df_html}</div>'
        html +=  cur_html
    html= f"""
    <div style="display:flex; gap:{gap}px; justify-content:{justify};">
        {html}
    </div>
    """
    display_html(html, raw=True)


def confusion_compare(fitted_clfs, data):
    """
    function to compare the confusion matrices resulted from each model

    Parmeters
    ---------
    fitted_clf: list
        names of the chosen classifier

    data: array of the dataset
    """
    classifiers = [clf[1] for clf in fitted_clfs]
    inputs, targets = data
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,8))

    for cls, ax in zip(classifiers, axes.flatten()):
        ConfusionMatrixDisplay.from_estimator(cls, inputs, targets, ax=ax, colorbar=False)
        ax.title.set_text(cls.__class__.__name__)
    plt.tight_layout()  
    plt.show()


def roc_compare(fitted_clfs, data):
    """
    function to compare the roc curves for the classifiers fitted
    """
    classifiers = [clf[1] for clf in fitted_clfs]
    inputs, targets = data
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

    for cls, ax in zip(classifiers, axes.flatten()):
        RocCurveDisplay.from_estimator(cls, inputs, targets, ax=ax)
        ax.title.set_text(cls.__class__.__name__)
    plt.tight_layout()  
    plt.show()

