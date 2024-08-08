# ClusterExplorer
This repository contains the code for ClusterExplorer, a novel explainability tool for black-box clustering pipelines. Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters.

## Explaining the results of clustering pipelines
Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters. We achieve this by reducing the problem to generalized frequent-itemsets mining (gFIM), where items correspond to explanation predicates, and itemset frequency indicates coverage. To enhance efficiency, we leverage inherent problem properties and implement attribute selection to further reduce computational costs.

## Source Code
The source code is located in the [`ClusterExplorer/src`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src) directory. This directory contains the following key components:

1. **Explainer**:[`explainer.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/explainer.py) Generates rule-based explanations for each cluster using frequent-itemsets mining.

2. **Frequent Itemset Mining**:[`gFIM.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/gFIM.py) Contains methods for frequent itemset mining..

3. **Clustering Rule Evaluation**:[`ScoreMetrics.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/ScoreMetrics.py) [`AnalyzeItemsets.py`](https://github.com/sarieltutay/ClusterExplorer/blob/main/src/AnalyzeItemsets.py) Provides methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.

4. **Binning Methods**:[`binning_methods.py`](https://github.com/yourusername/ClusterExplorer/blob/main/src/binning_methods.py) Contains methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.

## Experiment Datasets
Cluster-Explorer was evaluated using a diverse set of 98 clustering results obtained from various clustering pipelines and algorithms. The datasets used in these experiments were sourced from the UCI Machine Learning Repository and cover a wide range of data shapes and sizes.

### Datasets Overview
The datasets used in the experiments include:

| **Dataset**                             | **Rows**   | **Attributes** | **Link** |
|-----------------------------------------|------------|----------------|----------|
| Urban Land Cover                        | 168        | 148            | [Link](https://archive.ics.uci.edu/dataset/295/urban+land+cover) |
| DARWIN                                  | 174        | 451            | [Link](https://archive.ics.uci.edu/dataset/732/darwin) |
| Wine                                    | 178        | 13             | [Link](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Flags                                   | 194        | 30             | [Link](https://archive.ics.uci.edu/ml/datasets/Flags) |
| Parkinson Speech                        | 1,040      | 26             | [Link](https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with+Multiple+Types+of+Sound+Recordings) |
| Communities and Crime                   | 1,994      | 128            | [Link](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime) |
| Turkiye Student Evaluation              | 5,820      | 33             | [Link](https://archive.ics.uci.edu/ml/datasets/Turkiye+Student+Evaluation) |
| In-vehicle Coupon Recommendation        | 12,684     | 23             | [Link](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation) |
| Human Activity Recognition              | 10,299     | 561            | [Link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) |
| Quality Assessment of Digital Colposcopies | 30,000  | 23             | [Link](https://archive.ics.uci.edu/ml/datasets/Quality+Assessment+of+Digital+Colposcopies) |
| RT-IoT2022                              | 123,117    | 85             | [Link](https://archive.ics.uci.edu/ml/datasets/Real-Time+IoT+Network+Intrusion+Dataset+2022) |
| Gender by Name                          | 147,270    | 4              | [Link](https://archive.ics.uci.edu/ml/datasets/Gender+by+Name) |
| Multivariate Gait Data                  | 181,800    | 7              | [Link](https://archive.ics.uci.edu/ml/datasets/Multivariate+Gait+Data) |
| Wave Energy Converters                  | 288,000    | 49             | [Link](https://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters) |
| 3D Road Network                         | 434,874    | 4              | [Link](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29) |
| Year Prediction MSD                     | 515,345    | 90             | [Link](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) |
| Online Retail                           | 1,067,371  | 8              | [Link](https://archive.ics.uci.edu/ml/datasets/Online+Retail) |
| MetroPT-3 Dataset                       | 1,516,948  | 15             | [Link](https://archive.ics.uci.edu/ml/datasets/MetroPT-3) |
| Taxi Trajectory                         | 1,710,670  | 9              | [Link](https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge%2C+ECML+PKDD+2015) |

### Clustering Pipelines
The clustering results were generated using 16 different clustering pipelines, each combining various preprocessing steps and clustering algorithms (are located in [clustering_pipelines.py](https://github.com/yourusername/ClusterExplorer/blob/main/experiments/clustering_pipelines.py)). The preprocessing steps included standard scaling for numeric columns, one-hot encoding for categorical data, and dimensionality reduction using PCA. The clustering algorithms used were K-Means, DBSCAN, Birch, Spectral Clustering, and Affinity Propagation.





