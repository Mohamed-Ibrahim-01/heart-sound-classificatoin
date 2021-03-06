# Heart-Sound-Classification
- A machine learning project using python to build a system 
  to classify recorded heart sounds as either normal or abnormal 
  based on some selected features which are extracted from the
  sound files through a series of segmentation and feature extraction
  algorithms as a data preprocessing step before training
  different models to choose the best classifier based on
  some evaluation criteria.
 ## Installation
- You may need to install these libraries to run the notebook  
```python
pip install sklearn
pip install librosa
pip install scipy
pip install matplotlib
pip install xgboost
```
## Usage
- To run the clssification system the entry point for the application is:
- `HS_Classification_Main` notebook 
## Workflow
* First we load `the PhysioNet dataset 
[PhysionNet](https://physionet.org/content/challenge-2016/1.0.0/)
* Using the wavelet transform to make signal denoising. 
* Applying segmentation techniques to the loaded sound files.
* Going through the series of feature extraction methods.
  * These are the extracted featuers:

  Max_Amplitude|Dominant_Freq|Entropy|Mean|Median|STD|Kurtosis|Skewness|IQR|First_Percentile|Second_Percentile|Third_Percentile|MFCC1|MFCC2|MFCC3|MFCC4|MFCC5|MFCC6|MFCC7|MFCC8|MFCC9|MFCC10|MFCC11|MFCC12|MFCC13
  ---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- 
  
* Enhancing the performance by normalization, PCA, and feature selection algorithm.
   ```python
    X = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components= 0.99)
    var_threshold = VarianceThreshold(threshold = 0.011)
    var_threshold.fit(X)
    X_RLVF = var_threshold.transform(X)
    principal_X = pca.fit_transform(X)
    principal_X_RLVF = pca.fit_transform(X_RLVF)
    ```
    * The above code has reduced the number of features to 13 features.
* Making some exploration on the data to gain some insights.
  * Histogram of some features over the whole data set
  
  ![alt text](https://github.com/Mohamed-Ibrahim-01/heart-sound-classificatoin/blob/master/exploration.png)
  
* Building the data frame and training different models on the data.
* Examination of the results and choosing the best classifier to use.
    * We found that the best classifier is the Support Vector Machine ```SVM``` with training accuracy equals to
      ```94.2%``` and test accuracy equals to ```92.6%```  
