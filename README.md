# d-RITS - Disentangled Representation Learning for Medical Time Series

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

Since ***d-RITS*** performs 3 different tasks simultaneously, we evaluate the performance of each task separately.
#### Imputation Evaluation

[Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) and mean relative error (MAE and MRE) are used to evaluate ***d-RITS***'s imputation performance

| Imputation Methods          | MAE                          | MRE                          |
|-----------------------------|------------------------------|------------------------------|
| Mean                        | 1\.09                        | 1\.52                        |
| Forward Feeding             | 0\.71                        | 0\.99                        |
| Matrix Factorization        | 0\.47                        | 0\.66                        |
| KNN \(k=100\)               | 0\.54                        | 0\.75                        |
| **d-RITS**                  | **0\.42±0\.01**              | **0\.58±0\.01**              |


#### Classification Evaluation

The area under the precision-recall curve (AUPRC) is our main performance metric; not only is it suitable for evaluating our imbalanced data but also useful in clinical problem settings where high recall is preferred over high accuracy. 

|                          | Validation Average Precision (AUPRC) |
|--------------------------|-------------------------------------------------------------------------------|
| Models                   | Lung              | Liver             | GI                | Kidney            |
| Random Classifier        | 0\.425            | 0\.101            | 0\.420            | 0\.482            |
| Single Layer Perceptron  | 0\.595±0\.014     | 0\.301±0\.028     | 0\.456±0\.010     | 0\.759±0\.011     |
| Random Forest            | 0\.700±0\.002     | **0\.514±0\.019** | 0\.523±0\.005     | **0\.847±0\.007** |
| **d-RITS**               | 0\.711±0\.004     | 0\.459±0\.015     | **0\.530±0\.001** | 0\.822±0\.033     |
| LR based on **d-RITS**   | **0\.713±0\.015** | 0\.449±0\.011     | **0\.530±0\.002** | 0\.824±0\.029     |


#### Disentanglement Evaluation

One of the benefits of learning a disentangled representation is that it makes further processing and learning much easier. A well-disentangled representation produces features that specialize in specific label predictions. When we fit a very simple logistic regression model to our disentangled data, we get clean and comparable performance as shown in the previous plot.