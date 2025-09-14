import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import signal
import time
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation, MeanShift, OPTICS,
    MiniBatchKMeans, Birch, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.datasets import (
    make_moons, make_blobs, make_circles, load_iris, load_wine, 
    load_breast_cancer, make_classification
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
)
from scipy.spatial.distance import cdist, pdist
from scipy import stats
import psutil
import memory_profiler

# Import for KMedoids and HDBSCAN
try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("Warning: sklearn_extra not available. KMedoids will be skipped.")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. HDBSCAN will be skipped.")

# Global variables
results_directory = None
all_results = []
detailed_results = {}

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, handler)

def create_results_directory():
    """Create a timestamped directory for storing all results."""
    global results_directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_directory = Path(f"clustering_results_{timestamp}")
    
    # Create subdirectories
    subdirs = [
        'datasets', 'metrics', 'visualizations', 'statistical_analysis',
        'convergence_plots', 'detailed_results', 'summary_reports'
    ]
    
    for subdir in subdirs:
        (results_directory / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be stored in: {results_directory}")
    return results_directory

def save_results(data, filename, subdir='detailed_results', format='json'):
    """Save results to the appropriate subdirectory."""
    filepath = results_directory / subdir / f"{filename}.{format}"
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    
    print(f"Saved: {filepath}")

# Origin Initialization Method
def origin_initialization(X, k):
    """
    Initialize centroids by selecting points closest to the origin.
    This implements the Average Distance Initialization method from the OriginScale
    paper, which selects the k data points that are closest to the origin (0,0)
    as the initial centroids.
    """
    distances = np.linalg.norm(X, axis=1)
    sorted_indices = np.argsort(distances)
    centroids = X[sorted_indices[:k]]
    return centroids

# OriginScale Clustering Algorithm
class OriginScale:
    def __init__(self, n_clusters=3, tol=1e-4, max_iter=300):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None 

    def _euclidean_distance(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _average_distance_initialization(self, X):
        distances = np.linalg.norm(X, axis=1)
        sorted_indices = np.argsort(distances)
        return X[sorted_indices[:self.n_clusters]]

    def _assign_labels(self, X):
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            centroids.append(
                np.mean(cluster_points, axis=0) if len(cluster_points) > 0 else X[np.random.randint(X.shape[0])])
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._average_distance_initialization(X)
        prev_centroids = np.zeros_like(self.centroids)
        for _ in range(self.max_iter):
            labels = self._assign_labels(X)
            self.centroids = self._update_centroids(X, labels)
            if np.linalg.norm(self.centroids - prev_centroids) < self.tol:
                break
            prev_centroids = np.copy(self.centroids)
        self.labels_ = labels
        return self

    def predict(self, X):
        """Predict cluster labels for new data."""
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

def calculate_comprehensive_metrics(X, labels, true_labels=None):
    """Calculate comprehensive clustering evaluation metrics."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    k = len(unique_labels)
    
    # Filter noise points
    X_filtered = X[labels != -1]
    labels_filtered = labels[labels != -1]
    unique_labels_filtered = np.unique(labels_filtered)
    k_filtered = len(unique_labels_filtered)
    
    metrics = {
        "n_clusters": k,
        "n_clusters_filtered": k_filtered,
        "n_noise_points": np.sum(labels == -1),
        "noise_ratio": np.sum(labels == -1) / len(labels)
    }
    
    if k_filtered < 2:
        # Set all metrics to None if insufficient clusters
        metrics.update({
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
            "ssw": None,
            "sst": None,
            "ssb": None,
            "ssb_sst_ratio": None,
            "dunn_index": None,
            "xie_beni_index": None,
            "adjusted_rand_score": None,
            "normalized_mutual_info": None,
            "adjusted_mutual_info": None
        })
        return metrics
    
    # Calculate cluster centers
    cluster_centers = np.array([
        X_filtered[labels_filtered == i].mean(axis=0) 
        for i in unique_labels_filtered
    ])
    
    # Internal validation metrics
    try:
        metrics["silhouette_score"] = silhouette_score(X_filtered, labels_filtered)
    except:
        metrics["silhouette_score"] = None
        
    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_filtered, labels_filtered)
    except:
        metrics["calinski_harabasz_score"] = None
        
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(X_filtered, labels_filtered)
    except:
        metrics["davies_bouldin_score"] = None
    
    # Calculate SSW, SST, SSB
    SSW = 0
    for idx, cluster_label in enumerate(unique_labels_filtered):
        cluster_points = X_filtered[labels_filtered == cluster_label]
        if len(cluster_points) > 0:
            SSW += np.sum((cluster_points - cluster_centers[idx]) ** 2)
    
    overall_mean = np.mean(X_filtered, axis=0)
    SST = np.sum((X_filtered - overall_mean) ** 2)
    SSB = SST - SSW
    
    metrics.update({
        "ssw": SSW,
        "sst": SST,
        "ssb": SSB,
        "ssb_sst_ratio": SSB / SST if SST != 0 else None
    })
    
    # Dunn Index
    if k_filtered > 1:
        min_intercluster_dist = float('inf')
        for i in range(k_filtered):
            for j in range(i + 1, k_filtered):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                if dist < min_intercluster_dist:
                    min_intercluster_dist = dist
        
        max_intracluster_dist = 0.0
        for idx, cluster_label in enumerate(unique_labels_filtered):
            cluster_points = X_filtered[labels_filtered == cluster_label]
            if len(cluster_points) > 1:
                intra_dists = pdist(cluster_points, metric='euclidean')
                if len(intra_dists) > 0:
                    current_max_intra = np.max(intra_dists)
                    if current_max_intra > max_intracluster_dist:
                        max_intracluster_dist = current_max_intra
        
        metrics["dunn_index"] = (min_intercluster_dist / max_intracluster_dist 
                               if max_intracluster_dist != 0 else None)
    else:
        metrics["dunn_index"] = None
    
    # Xie-Beni Index
    if k_filtered > 1:
        min_dist_between_clusters = np.min(
            cdist(cluster_centers, cluster_centers)[~np.eye(k_filtered, dtype=bool)]
        )
        metrics["xie_beni_index"] = (SSW / (len(X_filtered) * min_dist_between_clusters) 
                                   if min_dist_between_clusters != 0 else None)
    else:
        metrics["xie_beni_index"] = None
    
    # External validation metrics (if true labels available)
    if true_labels is not None:
        try:
            metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, labels)
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(true_labels, labels)
            metrics["adjusted_mutual_info"] = adjusted_mutual_info_score(true_labels, labels)
        except:
            metrics["adjusted_rand_score"] = None
            metrics["normalized_mutual_info"] = None
            metrics["adjusted_mutual_info"] = None
    
    return metrics

def measure_memory_usage(func):
    """Decorator to measure memory usage of a function."""
    def wrapper(*args, **kwargs):
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        return result, memory_used
    return wrapper

@measure_memory_usage
def run_clustering_algorithm(clustering_func, X, **kwargs):
    """Run clustering algorithm with memory monitoring."""
    return clustering_func(X, **kwargs)

def run_with_timeout_and_monitoring(clustering_func, X, name, timeout_seconds=120, 
                                  dataset_name="unknown", true_labels=None, **kwargs):
    """Enhanced function to run clustering with comprehensive monitoring."""
    print(f"\nRunning {name} on {dataset_name}...")
    
    result = {
        "algorithm": name,
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "execution_time": None,
        "memory_usage": None,
        "timeout": False,
        "error": None,
        "labels": None
    }
    
    signal.alarm(timeout_seconds)
    
    try:
        start_time = time.time()
        
        if name == "OriginScale":
            # Special handling for OriginScale without convergence tracking
            model = clustering_func(X, **kwargs)
            labels = model.labels_
            memory_used = 0  # Simplified for OriginScale
        else:
            labels, memory_used = run_clustering_algorithm(clustering_func, X, **kwargs)
        
        execution_time = time.time() - start_time
        signal.alarm(0)
        
        result.update({
            "execution_time": execution_time,
            "memory_usage": memory_used,
            "labels": labels.tolist() if labels is not None else None
        })
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(X, labels, true_labels)
        result.update(metrics)
        
        print(f"✓ {name} completed in {execution_time:.4f}s")
        if metrics["silhouette_score"] is not None:
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        
    except TimeoutException:
        result["timeout"] = True
        result["error"] = "Timeout"
        print(f"✗ {name} timed out after {timeout_seconds}s")
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ {name} failed: {e}")
    finally:
        signal.alarm(0)
    
    all_results.append(result)
    return result

# Clustering algorithm implementations
def run_origin_scale(X, n_clusters=2):
    model = OriginScale(n_clusters=n_clusters)
    return model.fit(X)

def run_kmeans(X, n_clusters=2):
    return KMeans(n_clusters=n_clusters, init="random", n_init=10, random_state=42).fit(X).labels_

def run_kmeans_plus(X, n_clusters=2):
    return KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42).fit(X).labels_

def run_dbscan(X, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)

def run_agglo(X, n_clusters=2):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X)

def run_affinity(X):
    try:
        af = AffinityPropagation(random_state=42, damping=0.9, max_iter=200)
        return af.fit(X).labels_
    except:
        return np.full(X.shape[0], -1)

def run_gmm(X, n_clusters=2):
    return GaussianMixture(n_components=n_clusters, random_state=42).fit(X).predict(X)

def run_mean_shift(X):
    from sklearn.cluster import estimate_bandwidth
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(1000, X.shape[0]), 
                                 random_state=42, n_jobs=-1)
    if bandwidth <= 0:
        bandwidth = 1.0
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    return ms.fit(X).labels_

def run_optics(X):
    return OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05, n_jobs=-1).fit_predict(X)

def run_minibatch_kmeans(X, n_clusters=2):
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, 
                         batch_size=min(256, X.shape[0]//4)).fit(X).labels_

def run_birch(X, n_clusters=2):
    return Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50).fit(X).labels_

def run_spectral_clustering(X, n_clusters=2):
    n_neighbors = min(10, X.shape[0]//4)
    return SpectralClustering(n_clusters=n_clusters, random_state=42, 
                            affinity='nearest_neighbors', n_neighbors=n_neighbors, 
                            assign_labels='kmeans').fit_predict(X)

def run_kmedoids(X, n_clusters=2):
    if not KMEDOIDS_AVAILABLE:
        return None
    return KMedoids(n_clusters=n_clusters, random_state=42, 
                   init='k-medoids++', method='pam').fit(X).labels_

def run_hdbscan(X):
    if not HDBSCAN_AVAILABLE:
        return None
    min_cluster_size = max(5, X.shape[0] // 100)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(X)

def generate_datasets():
    """Generate various synthetic and real datasets for comprehensive testing."""
    datasets = {}
    
    # Synthetic datasets
    print("Generating synthetic datasets...")
    
    # 1. Moons dataset
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)
    datasets['moons'] = {'X': X_moons, 'y': y_moons, 'n_clusters': 2}
    
    # 2. Blobs dataset
    X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, 
                                random_state=42)
    datasets['blobs'] = {'X': X_blobs, 'y': y_blobs, 'n_clusters': 4}
    
    # 3. Circles dataset
    X_circles, y_circles = make_circles(n_samples=1000, noise=0.1, factor=0.3, 
                                      random_state=42)
    datasets['circles'] = {'X': X_circles, 'y': y_circles, 'n_clusters': 2}
    
    # 4. Anisotropic blobs
    X_aniso, y_aniso = make_blobs(n_samples=1000, centers=3, random_state=42)
    transformation = [[0.8, 0.2], [0.2, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets['anisotropic'] = {'X': X_aniso, 'y': y_aniso, 'n_clusters': 3}
    
    # Real-world datasets
    print("Loading real-world datasets...")
    
    # 5. Iris dataset
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)
    datasets['iris'] = {'X': X_iris, 'y': iris.target, 'n_clusters': 3}
    
    # 6. Wine dataset
    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data)
    datasets['wine'] = {'X': X_wine, 'y': wine.target, 'n_clusters': 3}
    
    # 7. Breast Cancer dataset
    cancer = load_breast_cancer()
    X_cancer = StandardScaler().fit_transform(cancer.data)
    datasets['breast_cancer'] = {'X': X_cancer, 'y': cancer.target, 'n_clusters': 2}
    
    # Save datasets
    for name, data in datasets.items():
        save_results({
            'X': data['X'].tolist(),
            'y': data['y'].tolist(),
            'n_clusters': data['n_clusters'],
            'n_samples': data['X'].shape[0],
            'n_features': data['X'].shape[1]
        }, f"dataset_{name}", 'datasets')
    
    return datasets

def run_comprehensive_comparison():
    """Run comprehensive clustering algorithm comparison."""
    print("Starting comprehensive clustering algorithm comparison...")
    
    # Generate datasets
    datasets = generate_datasets()
    
    # Define algorithms to test
    algorithms = [
        ("OriginScale", run_origin_scale),
        ("K-Means (Random)", run_kmeans),
        ("K-Means++", run_kmeans_plus),
        ("Mini-Batch K-Means", run_minibatch_kmeans),
        ("DBSCAN", run_dbscan),
        ("Agglomerative", run_agglo),
        ("Affinity Propagation", run_affinity),
        ("GMM", run_gmm),
        ("Mean Shift", run_mean_shift),
        ("OPTICS", run_optics),
        ("BIRCH", run_birch),
        ("Spectral Clustering", run_spectral_clustering),
    ]
    
    # Add optional algorithms if available
    if KMEDOIDS_AVAILABLE:
        algorithms.append(("K-Medoids", run_kmedoids))
    if HDBSCAN_AVAILABLE:
        algorithms.append(("HDBSCAN", run_hdbscan))
    
    # Run experiments
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing on dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        X = dataset_info['X']
        y_true = dataset_info['y']
        n_clusters = dataset_info['n_clusters']
        
        for alg_name, alg_func in algorithms:
            # Determine parameters based on algorithm
            kwargs = {}
            if alg_name in ["OriginScale", "K-Means (Random)", "K-Means++", 
                          "Mini-Batch K-Means", "Agglomerative", "GMM", 
                          "BIRCH", "Spectral Clustering", "K-Medoids"]:
                kwargs['n_clusters'] = n_clusters
            elif alg_name == "DBSCAN":
                # Adaptive eps based on dataset
                if dataset_name in ['moons', 'circles']:
                    kwargs['eps'] = 0.3
                else:
                    kwargs['eps'] = 0.5
                kwargs['min_samples'] = max(5, X.shape[0] // 200)
            
            # Skip algorithms that returned None (not available)
            if alg_func is None:
                continue
                
            # Run algorithm
            result = run_with_timeout_and_monitoring(
                alg_func, X, alg_name, 
                timeout_seconds=300,
                dataset_name=dataset_name,
                true_labels=y_true,
                **kwargs
            )
            
            # Store detailed results
            if dataset_name not in detailed_results:
                detailed_results[dataset_name] = {}
            detailed_results[dataset_name][alg_name] = result

def create_visualizations():
    """Create comprehensive visualizations of results."""
    print("\nCreating visualizations...")
    
    # 1. Performance comparison plots
    df_results = pd.DataFrame(all_results)
    
    # Filter successful runs
    df_success = df_results[
        (df_results['timeout'] == False) & 
        (df_results['error'].isna()) &
        (df_results['silhouette_score'].notna())
    ].copy()
    
    if len(df_success) > 0:
        # Execution time vs Silhouette score
        plt.figure(figsize=(15, 10))
        datasets_list = df_success['dataset'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets_list)))
        
        for i, dataset in enumerate(datasets_list):
            df_dataset = df_success[df_success['dataset'] == dataset]
            plt.scatter(df_dataset['execution_time'], df_dataset['silhouette_score'],
                       c=[colors[i]], label=dataset, s=100, alpha=0.7)
            
            # Annotate points
            for _, row in df_dataset.iterrows():
                plt.annotate(row['algorithm'], 
                           (row['execution_time'], row['silhouette_score']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        plt.xscale('log')
        plt.xlabel('Execution Time (seconds, log scale)')
        plt.ylabel('Silhouette Score')
        plt.title('Clustering Algorithm Performance: Execution Time vs Silhouette Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_directory / 'visualizations' / 'performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap of silhouette scores
        pivot_silhouette = df_success.pivot(index='algorithm', columns='dataset', 
                                          values='silhouette_score')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_silhouette, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Silhouette Scores Across Algorithms and Datasets')
        plt.tight_layout()
        plt.savefig(results_directory / 'visualizations' / 'silhouette_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Memory usage comparison
        df_memory = df_success[df_success['memory_usage'].notna()]
        if len(df_memory) > 0:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_memory, x='algorithm', y='memory_usage')
            plt.xticks(rotation=45)
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Comparison Across Algorithms')
            plt.tight_layout()
            plt.savefig(results_directory / 'visualizations' / 'memory_usage.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def perform_statistical_analysis():
    """Perform statistical analysis of results."""
    print("\nPerforming statistical analysis...")
    
    df_results = pd.DataFrame(all_results)
    df_success = df_results[
        (df_results['timeout'] == False) & 
        (df_results['error'].isna()) &
        (df_results['silhouette_score'].notna())
    ].copy()
    
    if len(df_success) == 0:
        print("No successful results for statistical analysis")
        return
    
    # Statistical summary
    summary_stats = {}
    
    metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 
              'execution_time', 'memory_usage']
    
    for metric in metrics:
        if metric in df_success.columns:
            metric_data = df_success.groupby('algorithm')[metric].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(4)
            summary_stats[metric] = metric_data.to_dict()
    
    # Save statistical summary
    save_results(summary_stats, 'statistical_summary', 'statistical_analysis')
    
    # Pairwise comparisons (if multiple datasets available)
    if len(df_success['dataset'].unique()) > 1:
        # Friedman test for multiple algorithms across datasets
        algorithms = df_success['algorithm'].unique()
        datasets = df_success['dataset'].unique()
        
        # Prepare data for Friedman test
        silhouette_matrix = []
        for dataset in datasets:
            dataset_scores = []
            for algorithm in algorithms:
                scores = df_success[
                    (df_success['dataset'] == dataset) & 
                    (df_success['algorithm'] == algorithm)
                ]['silhouette_score'].values
                if len(scores) > 0:
                    dataset_scores.append(scores[0])
                else:
                    dataset_scores.append(None)
            silhouette_matrix.append(dataset_scores)
        
        # Convert to numpy array and remove None values
        silhouette_matrix = np.array(silhouette_matrix, dtype=float)
        
        # Save statistical results
        statistical_results = {
            'summary_statistics': summary_stats,
            'silhouette_matrix': silhouette_matrix.tolist(),
            'algorithms': algorithms.tolist(),
            'datasets': datasets.tolist()
        }
        
        save_results(statistical_results, 'detailed_statistical_analysis', 
                    'statistical_analysis')

def generate_summary_report():
    """Generate a comprehensive summary report."""
    print("\nGenerating summary report...")
    
    df_results = pd.DataFrame(all_results)
    
    # Calculate success rates
    success_rates = {}
    for algorithm in df_results['algorithm'].unique():
        alg_data = df_results[df_results['algorithm'] == algorithm]
        total_runs = len(alg_data)
        successful_runs = len(alg_data[
            (alg_data['timeout'] == False) & 
            (alg_data['error'].isna())
        ])
        success_rates[algorithm] = {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0
        }
    
    # Best performers by metric
    df_success = df_results[
        (df_results['timeout'] == False) & 
        (df_results['error'].isna()) &
        (df_results['silhouette_score'].notna())
    ].copy()
    
    best_performers = {}
    if len(df_success) > 0:
        best_performers = {
            'highest_silhouette': df_success.loc[df_success['silhouette_score'].idxmax()].to_dict(),
            'fastest_execution': df_success.loc[df_success['execution_time'].idxmin()].to_dict(),
            'lowest_memory': df_success.loc[df_success['memory_usage'].idxmin()].to_dict() if 'memory_usage' in df_success.columns else None,
            'highest_calinski_harabasz': df_success.loc[df_success['calinski_harabasz_score'].idxmax()].to_dict() if 'calinski_harabasz_score' in df_success.columns else None,
            'lowest_davies_bouldin': df_success.loc[df_success['davies_bouldin_score'].idxmin()].to_dict() if 'davies_bouldin_score' in df_success.columns else None
        }
    
    # OriginScale specific analysis
    originscale_results = df_results[df_results['algorithm'] == 'OriginScale']
    originscale_analysis = {
        'total_runs': len(originscale_results),
        'successful_runs': len(originscale_results[
            (originscale_results['timeout'] == False) & 
            (originscale_results['error'].isna())
        ]),
        'average_iterations': None,  # Removed convergence tracking
        'convergence_patterns': []   # Removed convergence tracking
    }
    
    # Summary report
    summary_report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_algorithms_tested': len(df_results['algorithm'].unique()),
            'total_datasets_tested': len(df_results['dataset'].unique()),
            'total_experiments': len(df_results)
        },
        'success_rates': success_rates,
        'best_performers': best_performers,
        'originscale_analysis': originscale_analysis,
        'overall_statistics': {
            'successful_experiments': len(df_success),
            'timeout_rate': len(df_results[df_results['timeout'] == True]) / len(df_results),
            'error_rate': len(df_results[df_results['error'].notna()]) / len(df_results)
        }
    }
    
    # Save summary report
    save_results(summary_report, 'comprehensive_summary_report', 'summary_reports')
    
    # Create a human-readable summary
    readable_summary = f"""
# Clustering Algorithm Comparison Summary Report

## Experiment Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Algorithms**: {len(df_results['algorithm'].unique())}
- **Total Datasets**: {len(df_results['dataset'].unique())}
- **Total Experiments**: {len(df_results)}
- **Successful Experiments**: {len(df_success)}

## Algorithm Success Rates
"""
    
    for algorithm, stats in success_rates.items():
        readable_summary += f"- **{algorithm}**: {stats['successful_runs']}/{stats['total_runs']} ({stats['success_rate']:.2%})\n"
    
    if len(df_success) > 0 and best_performers.get('highest_silhouette'):
        readable_summary += f"""
## Best Performers
- **Best Silhouette Score**: {best_performers['highest_silhouette']['algorithm']} on {best_performers['highest_silhouette']['dataset']} (Score: {best_performers['highest_silhouette']['silhouette_score']:.4f})
- **Fastest Algorithm**: {best_performers['fastest_execution']['algorithm']} on {best_performers['fastest_execution']['dataset']} ({best_performers['fastest_execution']['execution_time']:.4f}s)
"""
    
    # --- Dataset-wise metrics table ---
    readable_summary += "\n## Dataset-wise Metrics, Memory, and Execution Time\n"
    metrics_to_show = [
        'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
        'execution_time', 'memory_usage'
    ]
    for dataset in df_success['dataset'].unique():
        readable_summary += f"\n### Dataset: {dataset}\n"
        df_ds = df_success[df_success['dataset'] == dataset]
        # Table header
        readable_summary += "| Algorithm | " + " | ".join([m.replace('_', ' ').title() for m in metrics_to_show]) + " |\n"
        readable_summary += "|---" * (len(metrics_to_show)+1) + "|\n"
        for _, row in df_ds.iterrows():
            vals = []
            for m in metrics_to_show:
                v = row.get(m, None)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    vals.append('N/A')
                elif m in ['execution_time', 'memory_usage']:
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(f"{v:.4f}")
            readable_summary += f"| {row['algorithm']} | " + " | ".join(vals) + " |\n"
    # --- End dataset-wise metrics table ---

    readable_summary += f"""
## OriginScale Performance
- **Success Rate**: {originscale_analysis['successful_runs']}/{originscale_analysis['total_runs']} ({originscale_analysis['successful_runs']/originscale_analysis['total_runs']:.2%} if originscale_analysis['total_runs'] > 0 else 0)

## Files Generated
- Raw results: `detailed_results/`
- Visualizations: `visualizations/`
- Statistical analysis: `statistical_analysis/`
- Dataset information: `datasets/`
"""
    
    # Save readable summary
    with open(results_directory / 'summary_reports' / 'README.md', 'w') as f:
        f.write(readable_summary)
    
    return summary_report

def create_convergence_plots():
    """Create convergence plots for algorithms that support it."""
    print("\nCreating convergence plots...")
    
    # Focus on OriginScale convergence
    originscale_results = [
        r for r in all_results 
        if r['algorithm'] == 'OriginScale' and r.get('convergence_history') is not None
    ]
    
    if originscale_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, result in enumerate(originscale_results[:4]):  # Plot up to 4 datasets
            if i >= 4:
                break
                
            ax = axes[i]
            convergence_history = result['convergence_history']
            
            ax.plot(convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_title(f"OriginScale Convergence - {result['dataset'].title()}")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Inertia')
            ax.grid(True, alpha=0.3)
            
            # Add annotations
            final_inertia = convergence_history[-1] if convergence_history else 0
            ax.annotate(f'Final: {final_inertia:.2f}', 
                       xy=(len(convergence_history)-1, final_inertia),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(originscale_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(results_directory / 'convergence_plots' / 'originscale_convergence.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_detailed_dataset_visualizations():
    """Create detailed visualizations for each dataset."""
    print("\nCreating detailed dataset visualizations...")
    
    # Load datasets again for visualization
    datasets = generate_datasets()
    
    for dataset_name, dataset_info in datasets.items():
        if dataset_info['X'].shape[1] == 2:  # Only visualize 2D datasets
            X = dataset_info['X']
            y_true = dataset_info['y']
            
            # Get results for this dataset
            dataset_results = [r for r in all_results if r['dataset'] == dataset_name 
                             and r['labels'] is not None and not r['timeout'] and r['error'] is None]
            
            if not dataset_results:
                continue
            
            # Create subplot for each algorithm
            n_algorithms = len(dataset_results)
            cols = min(4, n_algorithms)
            rows = (n_algorithms + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_algorithms == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if n_algorithms == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, result in enumerate(dataset_results):
                ax = axes[i] if n_algorithms > 1 else axes[0]
                
                labels = np.array(result['labels'])
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    if label == -1:  # Noise points
                        ax.scatter(X[labels == label, 0], X[labels == label, 1], 
                                 c='black', marker='x', s=50, alpha=0.6, label='Noise')
                    else:
                        ax.scatter(X[labels == label, 0], X[labels == label, 1], 
                                 c=[colors[j]], s=50, alpha=0.7, label=f'Cluster {label}')
                
                # Format silhouette score
                silhouette_str = f"{result['silhouette_score']:.3f}" if result['silhouette_score'] is not None else 'N/A'
                ax.set_title(f"{result['algorithm']}\nSilhouette: {silhouette_str}")
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(dataset_results), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(results_directory / 'visualizations' / f'clustering_results_{dataset_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to run the comprehensive clustering comparison."""
    print("="*80)
    print("COMPREHENSIVE CLUSTERING ALGORITHM COMPARISON")
    print("="*80)
    
    # Create results directory
    create_results_directory()
    
    try:
        # Run comprehensive comparison
        run_comprehensive_comparison()
        
        # Create visualizations
        create_visualizations()
        create_convergence_plots()
        create_detailed_dataset_visualizations()
        
        # Perform statistical analysis
        perform_statistical_analysis()
        
        # Generate summary report
        summary_report = generate_summary_report()
        
        # Save all results
        save_results(all_results, 'all_results_complete', 'detailed_results')
        save_results(detailed_results, 'detailed_results_by_dataset', 'detailed_results')
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved in: {results_directory}")
        print(f"Total experiments run: {len(all_results)}")
        print(f"Check the summary report at: {results_directory}/summary_reports/README.md")
        
        # Display quick summary
        df_results = pd.DataFrame(all_results)
        successful_results = df_results[
            (df_results['timeout'] == False) & 
            (df_results['error'].isna())
        ]
        
        print(f"\nQuick Summary:")
        print(f"- Successful runs: {len(successful_results)}/{len(all_results)}")
        print(f"- Algorithms tested: {len(df_results['algorithm'].unique())}")
        print(f"- Datasets tested: {len(df_results['dataset'].unique())}")
        
        if len(successful_results) > 0:
            best_silhouette = successful_results.loc[successful_results['silhouette_score'].idxmax()]
            print(f"- Best Silhouette Score: {best_silhouette['silhouette_score']:.4f} ({best_silhouette['algorithm']} on {best_silhouette['dataset']})")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        if all_results:
            save_results(all_results, 'partial_results_on_error', 'detailed_results')
            print(f"Partial results saved due to error.")

if __name__ == "__main__":
    main()
