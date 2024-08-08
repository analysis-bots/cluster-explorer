# ClusterExplorer
This repository contains the code for ClusterExplorer, a novel explainability tool for black-box clustering pipelines. Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters.

## Explaining the results of clustering pipelines
Our approach formulates the explanation of clusters as the identification of concise conjunctions of predicates that maximize the coverage of the cluster's data points while minimizing separation from other clusters. We achieve this by reducing the problem to generalized frequent-itemsets mining (gFIM), where items correspond to explanation predicates, and itemset frequency indicates coverage. To enhance efficiency, we leverage inherent problem properties and implement attribute selection to further reduce computational costs.

## Source Code
The source code is located in the [`cluster-explorer/src`](https://github.com/analysis-bots/cluster-explorer/tree/main/src) directory. This directory contains the following key components:

1. **Explainer**:[`explainer.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/src/explainer.py) Generates rule-based explanations for each cluster using frequent-itemsets mining.

2. **Frequent Itemset Mining**:[`gFIM.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/src/gFIM.py) Contains methods for frequent itemset mining..

3. **Clustering Rule Evaluation**:[`ScoreMetrics.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/src/ScoreMetrics.py) [`AnalyzeItemsets.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/src/AnalyzeItemsets.py) Provides methods to evaluate and summarize the quality of clustering rules based on metrics such as separation error, coverage, and conciseness.

4. **Binning Methods**:[`binning_methods.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/src/binning_methods.py) Contains methods for binning numeric attributes, including equal width, equal frequency, decision tree-based, and multiclass optimal binning techniques.

## Experiment Datasets
Cluster-Explorer was evaluated using a diverse set of 98 clustering results obtained from various clustering pipelines and algorithms. The datasets used in these experiments were sourced from the UCI Machine Learning Repository and cover a wide range of data shapes and sizes.

### Datasets Overview
The datasets used in the experiments include:

| **Dataset**                             | **Rows**   | **Attributes** | **Link** |
|-----------------------------------------|------------|----------------|----------|
| Urban Land Cover                        | 168        | 148            | [Link](https://archive.ics.uci.edu/dataset/295/urban+land+cover) |
| DARWIN                                  | 174        | 451            | [Link](https://archive.ics.uci.edu/dataset/732/darwin) |
| Wine                                    | 178        | 13             | [Link](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Flags                                   | 194        | 30             | [Link](https://archive.ics.uci.edu/dataset/40/flags) |
| Parkinson Speech                        | 1,040      | 26             | [Link](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings) |
| Communities and Crime                   | 1,994      | 128            | [Link](https://archive.ics.uci.edu/dataset/183/communities+and+crime) |
| Turkiye Student Evaluation              | 5,820      | 33             | [Link](https://archive.ics.uci.edu/dataset/262/turkiye+student+evaluation) |
| In-vehicle Coupon Recommendation        | 12,684     | 23             | [Link](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation) |
| Human Activity Recognition              | 10,299     | 561            | [Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| Quality Assessment of Digital Colposcopies | 30,000  | 23             | [Link](https://archive.ics.uci.edu/dataset/384/quality+assessment+of+digital+colposcopies) |
| RT-IoT2022                              | 123,117    | 85             | [Link](https://archive.ics.uci.edu/dataset/942/rt-iot2022) |
| Gender by Name                          | 147,270    | 4              | [Link](https://archive.ics.uci.edu/dataset/591/gender+by+name) |
| Multivariate Gait Data                  | 181,800    | 7              | [Link](https://archive.ics.uci.edu/dataset/760/multivariate+gait+data) |
| Wave Energy Converters                  | 288,000    | 49             | [Link](https://archive.ics.uci.edu/dataset/494/wave+energy+converters) |
| 3D Road Network                         | 434,874    | 4              | [Link](https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark) |
| Year Prediction MSD                     | 515,345    | 90             | [Link](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) |
| Online Retail                           | 1,067,371  | 8              | [Link](https://archive.ics.uci.edu/dataset/352/online+retail) |
| MetroPT-3 Dataset                       | 1,516,948  | 15             | [Link](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) |
| Taxi Trajectory                         | 1,710,670  | 9              | [Link](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015) |

### Clustering Pipelines
The clustering results were generated using 16 different clustering pipelines, each combining various preprocessing steps and clustering algorithms (are located in [`clustering_pipelines.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/experiments/clustering_pipelines.py)). The preprocessing steps included standard scaling for numeric columns, one-hot encoding for categorical data, and dimensionality reduction using PCA. The clustering algorithms used were K-Means, DBSCAN, Birch, Spectral Clustering, and Affinity Propagation.

To use this, you need to provide the datasets folder (first save the datasets in this folder) and the folder to save the pipelines results.

### Running the Experiments
For running the experiments (located in [`cluster-explorer/experiments`](https://github.com/analysis-bots/cluster-explorer/tree/main/experiments)), you need to provide the folder of the pipelines result for [`BaselinesExperiment.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/experiments/BaselinesExperiment.py). The results will be saved in [`cluster-explorer/experiments`](https://github.com/analysis-bots/cluster-explorer/tree/main/experiments))

## Additional Experiments
This folder contains information about our attribute-selection optimization on both the explanation quality and running times.
For running the experiments (located in [`cluster-explorer/additional_experiments`](https://github.com/analysis-bots/cluster-explorer/tree/main/additional_experiments)), you need to provide the folder of the pipelines result for [`P_ValueExperiment.py`](https://github.com/analysis-bots/cluster-explorer/tree/main/additional_experiments/P_ValueExperiment.py). The results will be saved in [`cluster-explorer/additional_experiments`](https://github.com/analysis-bots/cluster-explorer/tree/main/additional_experiments))

## Use Cases and Examples
An example of a simple use case is provided in the [example_notebook.ipynb](https://github.com/analysis-bots/cluster-explorer/tree/main/example_notebook.ipynb) file. In this notebook we generate an explanation rules set from the 'Wine' dataset. For each cluster, the ClusterExplorer generates a set of rules that explain the common properties of the wine samples within that cluster. These rules help in understanding why certain samples are grouped together and what distinguishes one cluster from another.






