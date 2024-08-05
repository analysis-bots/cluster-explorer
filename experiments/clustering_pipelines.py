import os
import pandas as pd
import traceback
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV
import numpy as np


class OptimalClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, max_clusters=10, algorithm='kmeans'):
        self.max_clusters = max_clusters
        self.algorithm = algorithm

    def fit(self, X, y=None):
        if self.algorithm == 'kmeans':
            self.best_model_ = self._find_optimal_kmeans(X)
        elif self.algorithm == 'birch':
            self.best_model_ = self._find_optimal_birch(X)
        elif self.algorithm == 'spectral':
            self.best_model_ = self._find_optimal_spectral(X)
        elif self.algorithm == 'affinity':
            self.best_model_ = AffinityPropagation()
            self.best_model_.fit(X)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def _find_optimal_kmeans(self, X):
        param_grid = {'n_clusters': range(2, self.max_clusters + 1)}
        grid_search = GridSearchCV(KMeans(), param_grid, cv=5, scoring='silhouette')
        grid_search.fit(X)
        return grid_search.best_estimator_

    def _find_optimal_birch(self, X):
        param_grid = {'n_clusters': range(2, self.max_clusters + 1)}
        grid_search = GridSearchCV(Birch(), param_grid, cv=5, scoring='silhouette')
        grid_search.fit(X)
        return grid_search.best_estimator_

    def _find_optimal_spectral(self, X):
        param_grid = {'n_clusters': range(2, self.max_clusters + 1)}
        grid_search = GridSearchCV(SpectralClustering(), param_grid, cv=5, scoring='silhouette')
        grid_search.fit(X)
        return grid_search.best_estimator_

def pipelines_generator_new(dataset):
    numeric_features = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()

    pipelines = []

    def create_pipeline(scaling, onehot, pca, algorithm):
        steps = []

        if scaling:
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            steps.append(
                ('num_transform', ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])))

        if onehot:
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
            steps.append(('cat_transform',
                          ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])))

        if pca:
            steps.append(('pca', PCA(n_components=2)))

        if algorithm == 'K-Means':
            steps.append(('cluster', OptimalClusterer(max_clusters=8, algorithm='kmeans')))
        elif algorithm == 'DBSCAN':
            steps.append(('cluster', DBSCAN(eps=0.5)))
        elif algorithm == 'Birch':
            steps.append(('cluster', OptimalClusterer(max_clusters=8, algorithm='birch')))
        elif algorithm == 'Spectral':
            steps.append(('cluster', OptimalClusterer(max_clusters=8, algorithm='spectral')))
        elif algorithm == 'Affinity Propagation':
            steps.append(('cluster', AffinityPropagation()))

        return Pipeline(steps=steps)

    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=True, algorithm='K-Means'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=False, algorithm='K-Means'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=True, algorithm='K-Means'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=False, algorithm='K-Means'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=True, algorithm='DBSCAN'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=False, algorithm='DBSCAN'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=True, algorithm='DBSCAN'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=False, algorithm='DBSCAN'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=True, algorithm='Birch'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=False, algorithm='Birch'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=True, algorithm='Birch'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=False, algorithm='Birch'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=True, algorithm='Spectral'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=True, algorithm='Spectral'))
    pipelines.append(create_pipeline(scaling=True, onehot=True, pca=True, algorithm='Affinity Propagation'))
    pipelines.append(create_pipeline(scaling=False, onehot=True, pca=True, algorithm='Affinity Propagation'))

    return pipelines


def clustering_pipelines(datasets_folder_name, dataset_path_results):
    datasets_files = os.listdir(datasets_folder_name)
    for dataset_file in datasets_files:
        try:
            dataset_path = os.path.join(datasets_folder_name, dataset_file)
            name = os.path.splitext(os.path.basename(dataset_path))[0]
            dataset = pd.read_csv(dataset_path, delimiter=",", on_bad_lines='skip', header=0)
            dataset = dataset.dropna()
            pipelines = pipelines_generator_new(dataset.copy())
            clusters_columns = pd.DataFrame()

            for pipeline in pipelines:
                try:
                    # Extract the steps' names from the pipeline
                    pipeline_steps = [step[0] for step in pipeline.steps]
                    print(pipeline_steps)

                    # Fit the pipeline on the dataset
                    clusters = pipeline.fit_predict(dataset)

                    # Check silhouette score
                    if len(set(clusters)) > 1:
                        silhouette_avg = silhouette_score(dataset, clusters)
                        if silhouette_avg > 0.1:
                            clusters_columns['_'.join(pipeline_steps)] = clusters

                except Exception as e:
                    print(f"Error processing {pipeline}: {str(e)}")
                    traceback.print_exc()
                    continue

            combined_df = pd.concat([dataset, clusters_columns], axis=1)
            path = os.path.join(dataset_path_results, f"{name}_clustered.csv")
            combined_df.to_csv(path, index=False)
        except Exception as e:
            # Log the error
            print(f"Error processing {dataset_file}: {str(e)}")
            traceback.print_exc()  # Print the traceback to understand the error

            # Continue to the next iteration even if there's an error
            continue
