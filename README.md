# d-RITS 
<!-- - Disentangled Representation Learning for Medical Time Series -->

> Learning *disentangled* representations of input data increases the interpretability 
> of deep learning models. Interpretable representations are particularly important in 
> clinical settings because of their ability to provide sensible explanations to medical 
> predictive tasks. ***d-RITS*** a supervised framework for generating sparse and
> disentangled representations for organ-related diagnostic predictions. It is capable
> of imputing missing data with low error rate and performing interpretable classifications.

## Data

The patient time series data comes from the MIMIC-III database: https://mimic.physionet.org/mimicdata/.  
The database contains deidentified health-related data of over 40,000 patients who stayed in the ICU of Beth Israel Deaconness Medical Center from 2001 to 2012.  

The data is available by request. See access instructions [here](https://mimic.physionet.org/gettingstarted/access/).  

#### Data Processing

Methods include but not limited to the following: 

* Outlier removal
* Time stamp resampling
* Patient filtering
* Normalization

#### Data Labeling

Labels were generated based on patient's ICD-9 codes into different vital organ categories. More information can be found [here](https://en.wikipedia.org/wiki/List_of_ICD-9_codes).

## Experimental Results

Since ***d-RITS*** performs 3 different tasks simultaneously, we evaluate the performance of each of the following tasks separately:  
* Imputation
* Classification
* Representation Learning

### Imputation Evaluation

[Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) and mean relative error (MAE and MRE) are used to evaluate ***d-RITS***'s imputation performance

| Imputation Methods          | MAE                          | MRE                          |
|-----------------------------|------------------------------|------------------------------|
| Mean                        | 1\.09                        | 1\.52                        |
| Forward Feeding             | 0\.71                        | 0\.99                        |
| Matrix Factorization        | 0\.47                        | 0\.66                        |
| KNN \(k=100\)               | 0\.54                        | 0\.75                        |
| **d-RITS**                  | **0\.42±0\.01**              | **0\.58±0\.01**              |


### Classification Evaluation

The area under the precision-recall curve (AUPRC) is our main performance metric; not only is it suitable for evaluating our imbalanced data but also useful in clinical problem settings where high recall is preferred over high accuracy. 

|  <td colspan=4>Validation Average Precision (AUPRC) ||||| 
|-|-|-|-|-
| Models                   | Lung              | Liver             | GI                | Kidney            |
| Random Classifier        | 0\.425            | 0\.101            | 0\.420            | 0\.482            |
| Single Layer Perceptron  | 0\.595±0\.014     | 0\.301±0\.028     | 0\.456±0\.010     | 0\.759±0\.011     |
| Random Forest            | 0\.700±0\.002     | **0\.514±0\.019** | 0\.523±0\.005     | **0\.847±0\.007** |
| **d-RITS**               | 0\.711±0\.004     | 0\.459±0\.015     | **0\.530±0\.001** | 0\.822±0\.033     |
| LR based on **d-RITS**   | **0\.713±0\.015** | 0\.449±0\.011     | **0\.530±0\.002** | 0\.824±0\.029     |
  
    
(a) Lung diagnosis prediction
![lung](https://github.com/amarisch/d-RITS/blob/main/images/prc/run_prc_plot_Lung_0.png)  
(b) Liver diagnosis prediction
![liver](https://github.com/amarisch/d-RITS/blob/main/images/prc/run_prc_plot_Liver_0.png)  
(c) GI diagnosis prediction
![gi](https://github.com/amarisch/d-RITS/blob/main/images/prc/run_prc_plot_GI_0.png)  
(d) Kidney diagnosis prediction
![kidney](https://github.com/amarisch/d-RITS/blob/main/images/prc/run_prc_plot_Kidney_0.png)  
  
### Disentanglement Evaluation

One of the benefits of learning a disentangled representation is that it makes further processing and learning much easier. A well-disentangled representation produces features that specialize in specific label predictions. Three novel methods we used to evaluate disentanglement includes:
* Simple Model Fitting: when we fit a very simple logistic regression model to our disentangled data, we get clean and comparable performance as shown in the previous plot.
* Mutual Information(MI): our learned features have high MI with their corresponding labels.
* Shapley Values: a way to uniformly measure feature importance, and our model only places importance on a selected few features.
* Principal Component Analysis: our learned features separate our data into distinct clusters in 2D space.

### Mutual Information
Mutual information(MI) helps us to measure the *degree of disentangling*. It measures the extent to which one random variable represents another. For instance, given two random variables Y and C, MI measures the decrease in uncertainty about Y if we have the knowledge of C.
  
The normalized mutual information of the latent representations can be used to indicate the degree of disentanglement. A better disentangling means that some of the learned features have a higher mutual information with some of the known factors.

| <td colspan=5>Normalized Mutual Information |||||
|-|-|-|-|-|
| <td colspan=4>Class Labels |
| Disentangled Features  | Lung       | Liver           | GI                 | Kidney             |
|  Lung     | **0\.366** | 0\.068          | 0\.117             | 0\.106             |
|  Liver    | 0\.071     | **0\.380**      | 0\.083             | 0\.073             |
|  GI       | 0\.105     | 0\.125          | **0\.235**         | 0\.206             |
|  Kidney   | 0\.099     | 0\.075          | 0\.172             | **0\.359**         |


### Shapley Values
  
No Regularization          |  With Regularization
:-------------------------:|:-------------------------:
![noreg](https://github.com/amarisch/d-RITS/blob/main/images/shap/shap_noreg.png)  |  ![reg](https://github.com/amarisch/d-RITS/blob/main/images/shap/shap_elas.png)


### Principal Component Analysis

PCA of the disentangled features allows us to examine how the features are representated in a two-dimensional space.

(a) Lung
![pcalung](https://github.com/amarisch/d-RITS/blob/main/images/pca/pca_run3_Lung.png)  
(b) Liver
![pcaliver](https://github.com/amarisch/d-RITS/blob/main/images/pca/pca_run3_Liver.png)  
(c) GI
![pcagi](https://github.com/amarisch/d-RITS/blob/main/images/pca/pca_run3_GI.png)  
(d) Kidney
![pcakidney](https://github.com/amarisch/d-RITS/blob/main/images/pca/pca_run3_Kidney.png)  