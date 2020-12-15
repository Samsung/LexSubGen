import logging
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Iterable, Union
from pathlib import Path

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score

from lexsubgen.utils.params import build_from_config_path

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

SIL_DISTANCE = "cosine"
NC_RANGE = range(2, 9)


class SubstituteClusterizer:
    def __init__(
        self,
        n_clusters: Union[int, str] = "maxsil=range(2, 9)",
        min_df: int = 1,
        use_tfidf: bool = False,
        linkage: str = "average",
        affinity: str = "cosine",
    ):
        """
        This class provides an interface (predict method) for clustering sets of substitutes

        Args:
            n_clusters: number of clusters or algorithm of selecting number of clusters
                "fix": clustering with a fixed number of clusters ("n_clusters")
                    For example: n_clusters = 4
                "maxsil": selecting optimal number of clusters by maximizing silhouette score
                    In this case n_clusters means that nc will be selected from range(2, n_clusters + 1)
                    For example:
                        n_clusters = "maxsil=4" means nc from range(2, 5)
                        n_clusters = "maxsil=range(*args)": select nc from range(*args)
                "maxsil+merging": selecting optimal number of clusters by maximizing silhouette score
                    But in this case vectors are first clustered into "n_clusters" clusters
                    and then outliers are merged until silhouette score becomes maximum
                    For example:
                        n_clusters = "maxsil+merging=4": first clustering in 4 nc, and then merge outliers from range(2, 5)
                        n_clusters = "maxsil+merging=range(*args)": select nc from range(*args)
            min_df: standard parameter of TfidfVectorizer and CountVectorizer
            use_tfidf: vectorization of substitutes using
                TfidfVectorizer if use_tfidf is True or CountVectorizer otherwise
            linkage: standard parameter of AgglomerativeClustering
            affinity: standard parameter of AgglomerativeClustering
        """
        self.n_clusters = n_clusters
        self.use_tfidf = use_tfidf
        self.linkage = linkage
        self.affinity = affinity
        self.min_df = min_df

        self.descriptor = {
            "Clusterizer": {
                "name": self.__class__.__name__,
                "n_clusters": n_clusters,
                "min_df": min_df,
                "use_tfidf": use_tfidf,
                "linkage": linkage,
                "affinity": affinity,
            }
        }

    @staticmethod
    def _convert_str_to_n_clusters(value: str) -> Tuple[str, Iterable]:
        """
        Parses n_cluster argument. See the examples.
        Args:
            value: string that contains a clustering mode and numbers of clusters
                that might be represented in different ways. See the examples.
        Returns:
            mode: "maxsil" or "maxsil-merging"
            range: range object
        Examples:
            "maxsil": return ("maxsil", range(2, 9))
            "maxsil+merging=4": return ("maxsil+merging", range(2, 5))
            "maxsil=range(1, 10, 1)": return ("maxsil", range(1, 10, 1))
        """

        # "maxsil" converts to ["maxsil"] or
        # "maxsil+merging=4" converts to ["maxsil+merging", "4"] or
        # "maxsil=range(1, 10, 1)" converts to ["maxsil", "range(1, 10, 1)"]
        splitted = value.split("=")
        if len(splitted) == 1:
            return splitted[0], NC_RANGE
        elif len(splitted) == 2:
            mode, nc_arg = splitted
            # convert "range(1, 10, 1)" to "1, 10, 1"
            range_args_str = re.findall(r"range\((.+)\)", nc_arg)
            if len(range_args_str) == 1:
                # convert "1, 10, 1" to ["1", "10", "1"]
                range_args = re.split(r"\s*,\s*", range_args_str[0].strip())
                # convert ["1", "10", "1"] to range(1, 10, 1)
                nc_range = range(*[int(arg) for arg in range_args])
            else:
                # convert "4" to range(2, 5)
                nc_range = range(2, int(nc_arg) + 1)
            return mode, nc_range
        else:
            raise ValueError(
                "Invalid value of n_clusters parameter: it has more than one '=' character. "
                "Check examples."
            )

    @property
    def n_clusters(self) -> Union[int, Iterable]:
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: Union[int, str]):
        if isinstance(value, int):
            self.nc_selection_mode, self._n_clusters = "fix", value
        elif isinstance(value, str):
            mode, nc = self._convert_str_to_n_clusters(value)
            self.nc_selection_mode, self._n_clusters = mode, nc
        else:
            raise TypeError(
                "Invalid type of n_clusters parameter. "
                "Available types: int and str. "
                f"Given value: {value}"
            )

    @classmethod
    def from_config(cls, config_path):
        """
        Method creates cls instance from given config_path.
        Args:
            config_path: Path to file with clusterizer config.
        Returns:
            clusterizer: Object that can do clustering.
        """
        clusterizer, _ = build_from_config_path(config_path)
        return clusterizer

    def perform_clustering(
        self, n_clusters: int, vectors: np.ndarray, memory=None
    ) -> np.ndarray:
        """
        TODO: Add docs!
        Args:
            n_clusters:
            vectors:

        Returns:

        """
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage,
            affinity=self.affinity,
            memory=memory,
            compute_full_tree=memory is not None,
        )

        clustering.fit(vectors)
        return clustering.labels_

    def _find_optimal_clusterization_by_maximizing_sil_score(
        self, vectors: np.ndarray, n_clusters_range: Iterable[int], memory=None
    ) -> Tuple[np.ndarray, float]:
        """
        TODO: Add docs!
        Args:
            vectors:
            n_clusters_range:

        Returns:

        """
        n_vectors, _ = vectors.shape
        n_clusters_range = [nc for nc in n_clusters_range if 2 <= nc <= n_vectors - 1]
        assert len(n_clusters_range) > 0, f"Numbers of clusters have invalid values"

        max_sil_score = None
        opt_labels = np.zeros(n_vectors, dtype=np.int32)

        for n_clusters in n_clusters_range:
            pred_labels = self.perform_clustering(n_clusters, vectors, memory)
            sil_score = silhouette_score(vectors, pred_labels, metric=SIL_DISTANCE)
            if max_sil_score is None or sil_score > max_sil_score:
                opt_labels, max_sil_score = pred_labels, sil_score
        return opt_labels, max_sil_score

    def _merge_outliers_to_maximize_sil_score(
        self,
        vectors: np.ndarray,
        n_clusters_range: Iterable[int], labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        TODO: Add docs!
        Args:
            vectors:
            n_clusters_range:
            labels:

        Returns:

        """
        opt_labels = labels
        initial_nc = len(set(opt_labels))
        if 2 <= initial_nc <= len(vectors) - 1:
            max_sil_score = silhouette_score(vectors, opt_labels, metric=SIL_DISTANCE)
        else:
            max_sil_score = None

        n_vectors, _ = vectors.shape
        n_clusters_range = [nc for nc in n_clusters_range if 2 <= nc <= n_vectors - 1]
        if len(n_clusters_range) == 0:
            raise ValueError(f"Numbers of clusters have invalid values")

        for n_clusters in n_clusters_range:
            if n_clusters > initial_nc:
                continue

            pred_labels = self._merge_outliers(n_clusters, labels, vectors)
            sil_score = silhouette_score(vectors, pred_labels, metric=SIL_DISTANCE)

            if max_sil_score is None or sil_score > max_sil_score:
                opt_labels, max_sil_score = pred_labels, sil_score

        return opt_labels, max_sil_score

    @staticmethod
    def _get_centroids(
        clusters_set: List, labels: np.ndarray, vectors: np.ndarray
    ) -> Dict[Any, np.ndarray]:
        """
        TODO: Add docs!
        """
        label2mask = defaultdict(list)
        for i, label in enumerate(labels):
            label2mask[label].append(i)
        centroids = dict()
        for label in set(clusters_set):
            # computing the centroid for a set of vectors with the same label
            centroids[label] = np.mean(vectors[label2mask[label]], axis=0)
        return centroids

    def _merge_outliers(
        self, target_nc: int, labels: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """
        Less frequent clusters are assumed to be outliers.
        Then outliers are merged with the closest cluster (non outlier).

        Args:
            target_nc: merging outliers until n_clusters becomes target_nc
            labels: result of performed clusterization
            vectors:

        Returns:
            merged_labels: cluster labels after merging outliers
        """

        # sorting labels by frequency
        sorted_clusters = Counter(labels).most_common()

        if len(sorted_clusters) <= target_nc:
            return labels

        regular_cl_labels = [label for label, _ in sorted_clusters[:target_nc]]
        outlier_cl_labels = {label for label, _ in sorted_clusters[target_nc:]}
        centroids = self._get_centroids(regular_cl_labels, labels, vectors)

        merged_labels = labels.copy()
        for i, label in enumerate(labels):
            if label not in outlier_cl_labels:
                continue
            # finding the closest cluster for each outlier
            cl_idx = np.argmin([
                distance.cosine(centroids[reg_l], vectors[i])
                for reg_l in regular_cl_labels
            ])
            merged_labels[i] = regular_cl_labels[int(cl_idx)]
        return merged_labels

    @staticmethod
    def _get_min_df(documents: List[List[str]]) -> int:
        """
        Finds a substitute that occurs in minimum number of documents.
        In this case document is a list of substitutes.

        Args:
            substitutes: list of list of substitutes

        Returns:
            min_count: number of documents containing this substitute
        """
        counter = Counter(s for substs in documents for s in set(substs))
        return min(
            max(counter[s] for s in set(substs))
            for substs in documents
        )

    def vectorize_documents(
        self,
        documents: List[List[str]]
    ) -> np.ndarray:
        # to avoid a case when we can get a zero vector we need to find correct min_df value
        min_df = min(self.min_df, self._get_min_df(documents))

        assert min_df > 0, f"Invalid min_df value: {min_df}"

        if self.use_tfidf:
            vectorizer = TfidfVectorizer(
                min_df=min_df, lowercase=False, tokenizer=lambda x: x
            )
        else:
            vectorizer = CountVectorizer(
                min_df=min_df, lowercase=False, tokenizer=lambda x: x
            )
        return vectorizer.fit_transform(documents).todense()

    def predict(
        self,
        documents: List[List[str]],
        memory=None
    ) -> np.ndarray:
        """
        Clusters bunch of documents into a fixed or automatically selected number of clusters

        Args:
            documents: each document is a list of words

        Returns:
            labels obtained after clusterization
        """
        vectors = self.vectorize_documents(documents)

        if self.nc_selection_mode == "fix":
            labels = self.perform_clustering(self.n_clusters, vectors)
        elif self.nc_selection_mode == "maxsil":
            labels, _ = self._find_optimal_clusterization_by_maximizing_sil_score(
                vectors, n_clusters_range=self.n_clusters, memory=memory
            )
        elif self.nc_selection_mode == "maxsil+merging":
            nc_list = list(self.n_clusters)
            labels = self.perform_clustering(max(nc_list), vectors)
            labels, _ = self._merge_outliers_to_maximize_sil_score(
                vectors, n_clusters_range=nc_list, labels=labels
            )
        else:
            raise ValueError(f"Unknown nc_selection mode: {self.nc_selection_mode}")
        return labels
