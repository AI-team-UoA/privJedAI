"""
Docstring for pprl.clustering
"""
from __future__ import annotations
import gc
from collections import defaultdict, deque
from time import time
from typing import Dict, List, Tuple
import cudf
import cugraph
import rmm
import cupy as cp
import numpy as np
from privjedai.encoded_data import BloomEncodedData
from privjedai.gpu.base_clustering import AbstractClustering

RANDOM_SEED = 42

class ConnectedComponentsClustering(AbstractClustering):
    """Creates the connected components of the graph. \
        Applied to graph created from entity matching. \
        Input graph consists of the entity ids (nodes) and the similarity scores (edges).
    """


    _method_name: str = "Connected Components Clustering"
    _method_short_name: str = "CCC"
    _method_info: str = "Gets equivalence clusters from the " + \
                    "transitive closure of the similarity graph."

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float


    def process(self, graph: tuple, encoded_data: BloomEncodedData,
            similarity_threshold: float = None) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """

        start_time = time()
        self.encoded_data = encoded_data
        self.similarity_threshold: float = similarity_threshold

        edges, weights = graph
        valid_edges : np.ndarray
        if self.similarity_threshold is not None:
            valid_edges, valid_weights = self._get_valid_edges_and_weights(graph)
            del valid_weights, edges, weights
            cp.get_default_memory_pool().free_all_blocks()
        else:
            valid_edges = edges

        resulting_clusters = self.scipi_connected_components(
            edges=valid_edges
        )

        self.execution_time = time() - start_time
        return resulting_clusters

    @staticmethod
    def scipi_connected_components(edges:np.array) -> list:
        """Scipy Connected Components Algorithm in the produced graph.

        Args:
            edges (np.array): Consists of the entity ids (nodes) and the similarity scores (edges).
        Returns:
            list: list of clusters
        """

        if len(edges) == 0:
            return []

        flat_nodes = edges.ravel()
        max_id = int(flat_nodes.max()) + 1
        degrees = cp.bincount(flat_nodes, minlength=max_id)
        deg_u = degrees[edges[:, 0]]
        deg_v = degrees[edges[:, 1]]
        mask = (deg_u == 1) & (deg_v == 1)
        del deg_u, deg_v, degrees
        cp.get_default_memory_pool().free_all_blocks()

        isolated_edges = edges[mask].get()

        del edges, mask
        cp.get_default_memory_pool().free_all_blocks()


        return isolated_edges.tolist()




class UniqueMappingClustering(AbstractClustering):
    """Prunes all edges with a weight lower than t, sorts the remaining ones in
        decreasing weight/similarity and iteratively forms a partition for
        the top-weighted pair as long as none of its entities has already
        been matched to some other.
    """
    _method_name: str = "Unique Mapping Clustering"
    _method_short_name: str = "UMC"
    _method_info: str = "Prunes all edges with a weight lower than t, \
            sorts the remaining ones in" + \
            "decreasing weight/similarity and iteratively forms a partition for" + \
            "the top-weighted pair as long as none of its entities has already" + \
            "been matched to some other."

    def __init__(self) -> None:
        """Unique Mapping Clustering Constructor

        Args:
            similarity_threshold (float, optional): Prunes all edges with a weight
                lower than this. Defaults to 0.1.
            data (Data): Dataset module.
        """
        super().__init__()
        self.similarity_threshold: float

    def _get_unique_edges(self, candidates: cp.array) -> cp.array:

        used = np.zeros(candidates.max() + 1, dtype=bool)
        unique_edge_indices = []
        for i, (u,v) in enumerate(candidates):
            if not used[u] and not used[v]:
                used[u] = True
                used[v] = True
                unique_edge_indices.append(i)

        return cp.asarray(candidates[unique_edge_indices])


    def process(self, graph: Tuple[np.array, np.array], encoded_data: BloomEncodedData,
            similarity_threshold: float = 0.1) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        self.similarity_threshold: float = similarity_threshold

        start_time = time()
        self.encoded_data = encoded_data

        edges, weights = graph
        valid_edges, valid_weights = self._get_valid_edges_and_weights(graph)
        del edges, weights

        valid_weights *= -1
        cp.get_default_memory_pool().free_all_blocks()

        keys_stack = cp.stack([valid_edges[:,1], valid_edges[:, 0], valid_weights])
        sort_indices = cp.lexsort(keys_stack)
        candidates = valid_edges[sort_indices]
        del keys_stack, sort_indices, valid_edges, valid_weights
        cp.get_default_memory_pool().free_all_blocks()

        candidates = candidates[:candidates.shape[0]*2].get()

        unique_edges = self._get_unique_edges(candidates)
        clusters = ConnectedComponentsClustering.scipi_connected_components(
            edges=unique_edges
        )
        self.execution_time = time() - start_time
        return clusters

class CenterClustering(AbstractClustering):
    """Implements the Center Clustering algorithm.
    Input comparisons (graph edges) are sorted in descending order of similarity.
    Pairs of entities connected by these edges form the basis of the updated graph.
    Entities are evaluated to determine if they will serve
    as a center of a future cluster or as its member.
    This evaluation is based on a comparison of their cumulative edge weights in the graph,
    normalized by the number of edges in which they are involved.
    Finally, the algorithm identifies connected components within the graph,
    using the previously defined centers as the focal points for forming clusters.
    """


    _method_name: str = "Center Clustering"
    _method_short_name: str = "CC"
    _method_info: str = "Implements the Center Clustering algorithm," + \
        "In essence, it keeps it defines if a node within an edge constitutes \
            a center or member of future clusters" + \
        " by normalized over the graph weight sum comparison"
    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold: float

    def _init_comparisons(self, graph: tuple):
        valid_edges, valid_weights = self._get_valid_edges_and_weights(graph)
        valid_weights *= -1

        keys_stack = cp.stack([valid_edges[:,1], valid_edges[:, 0], valid_weights])
        sort_indices = cp.lexsort(keys_stack)
        candidates = valid_edges[sort_indices]
        similarities = valid_weights[sort_indices]

        candidates = candidates[:valid_edges.shape[0]*2].get()
        similarities = similarities[:valid_edges.shape[0]*2].get()

        del keys_stack, sort_indices, valid_edges, valid_weights
        cp.get_default_memory_pool().free_all_blocks()

        flat_nodes = valid_edges.ravel()
        duplicated_weights = cp.repeat(valid_weights, 2)
        edges_weight = cp.bincount(flat_nodes, weights=duplicated_weights)
        edges_attached = cp.bincount(flat_nodes)
        del flat_nodes, duplicated_weights
        cp.get_default_memory_pool().free_all_blocks()

        return edges_weight, edges_attached, candidates


    def _create_clusters(self, edges_weight: cp.array, edges_attached: cp.array,
                        candidates: np.array):
        cluster_centers = set()
        cluster_members = set()
        indices_to_keep = []
        for i, (v1, v2) in enumerate(candidates):
            v1_is_center : bool = v1 in cluster_centers
            v2_is_center : bool = v2 in cluster_centers
            v1_is_member : bool = v1 in cluster_members
            v2_is_member : bool = v2 in cluster_members
            if not (v1_is_center or v2_is_center or v1_is_member or v2_is_member):
                cluster_centers.add(v1
                    if edges_weight[v1]/edges_attached[v1] > edges_weight[v2]/edges_attached[v2]
                    else v2)
                cluster_members.add(v1
                    if edges_weight[v1]/edges_attached[v1] <= edges_weight[v2]/edges_attached[v2]
                    else v2)
                indices_to_keep.append(i)
            elif ((v1_is_center and v2_is_center) or (v1_is_member and v2_is_member)):
                continue
            elif (v1_is_center and not v2_is_member):
                cluster_members.add(v2)
                indices_to_keep.append(i)
            elif (v2_is_center and not v1_is_member):
                cluster_members.add(v1)
                indices_to_keep.append(i)

        return ConnectedComponentsClustering.scipi_connected_components(
            edges=candidates[indices_to_keep]
        )


    def process(self, graph: Tuple[np.array, np.array], encoded_data: BloomEncodedData,
            similarity_threshold: float = 0.5) -> list:
        """
        Docstring for process

        :param self: Description
        :param graph: Description
        :type graph: Graph
        :param encoded_data: Description
        :type encoded_data: BloomEncodedData
        :param similarity_threshold: Description
        :type similarity_threshold: float
        :return: Description
        :rtype: list
        """
        start_time = time()
        self.similarity_threshold : float = similarity_threshold
        self.encoded_data = encoded_data

        edges_weight, edges_attached, candidates = self._init_comparisons(graph)

        clusters = self._create_clusters(edges_weight, edges_attached,
                        candidates)
        self.execution_time = time() - start_time
        return clusters

class _KiralyVectors:
    """Data class to hold the vectors
    used in the Kiraly MSM Approximate Clustering algorithm."""

    # men_candidates: np.array
    # key_women_candidates: np.array

    current_matches : np.array
    edge_score: np.array
    edge_index: np.array
    men_active: np.array
    is_bachelor: np.array
    is_uncertain: np.array
    fiances: np.array

    def __init__(self, candidates_len: int, men_len: int,
        women_len: int) -> None:
        self.men_active = np.full(candidates_len, True, bool)
        self.edge_score = np.full((men_len, women_len), -np.inf)
        self.edge_index = np.full((men_len, women_len), -1, dtype=int)
        self.is_bachelor = cp.asarray(np.full(men_len, False, bool))
        self.is_uncertain = np.full(men_len, False, bool)
        self.fiances = np.full(women_len, -1, np.int32)
        self.current_matches = np.full(men_len, -1, np.int32)

    def has_active(self, indices: np.array) -> bool:
        """Returns True if any of the candidates at the given indices is active."""
        return self.men_active[indices].any()

    def get_first_active_index(self, indices: np.array) -> int:
        """Returns the index of the first active candidate
        among the given indices, or -1 if none are active."""
        active_mask = self.men_active[indices]
        return indices[active_mask][0]

    def update_man(self, man: int, indices: np.array) -> None:
        """Updates the state of a man who has no active candidates."""
        if not self.is_bachelor[man]:
            self.is_bachelor[man] =True
            self.men_active[indices] = True


    def update_current_matches(self,
        man_score: float,
        man: int,
        woman: int,
        index: int,) -> None:
        """Updates the current matches based on a proposal from a
        man."""
        append_to_free_men = -1
        men_len = self.current_matches.shape[0]
        fiance = self.fiances[woman - men_len]
        if fiance == -1:
            self.current_matches[man] = woman
            self.edge_score[man, woman - men_len] = man_score
            self.edge_index[man, woman - men_len] = index
            self.fiances[woman - men_len] = man
        else:
            fiance_score = self.edge_score[fiance, woman - men_len]
            fiance_index = self.edge_index[fiance, woman - men_len]
            if self.is_uncertain[fiance] or man_score > fiance_score:
                append_to_free_men = fiance
                self.current_matches[fiance] = -1
                self.current_matches[man] = woman
                self.edge_score[man, woman - men_len] = man_score
                self.edge_index[man, woman - men_len] = index
                self.fiances[woman - men_len] = man
                if not self.is_uncertain[fiance]:
                    self.men_active[fiance_index] = False
            else:
                append_to_free_men = man
                self.men_active[index] = False

        return append_to_free_men

class KiralyMSMApproximateClustering(AbstractClustering):
    """Implements the Kiraly MSM Approximate Clustering algorithm.
    Implements the so-called "New Algorithm"
    by Zoltan Kiraly 2013, which is a 3/2-approximation
    to the Maximum Stable Marriage (MSM) problem.
    The pairs resulting from the approximation
    of the stable relationships are translated into a graph,
    whose connected components we retain.
    """

    _method_name: str = "Kiraly MSM Approximate Clustering"
    _method_short_name: str = "KMAC"
    _method_info: str = "Ιmplements the Kiraly MSM Approximate Clustering algorithm," + \
        "In essence, it is a 3/2-approximation to the Maximum Stable Marriage (MSM) problem."
    key_women_candidates: np.array

    def __init__(self) -> None:
        super().__init__()
        self.similarity_threshold : float
        self.key_women_candidates: np.array


    def _get_sorted_edges_and_similarities(self, graph: tuple) -> Tuple[np.array, np.array]:
        valid_edges, valid_weights = graph
        sorted_indexes = cp.argsort(valid_weights)[::-1]
        sorted_edges_gpu = valid_edges[sorted_indexes]
        similarities_gpu =  valid_weights[sorted_indexes]
        del sorted_indexes
        cp.get_default_memory_pool().free_all_blocks()

        sorted_edges = sorted_edges_gpu.get()
        del sorted_edges_gpu
        cp.get_default_memory_pool().free_all_blocks()
        similarities = similarities_gpu.get()
        del similarities_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return sorted_edges, similarities

    def _get_current_matches(self,
            free_men: deque,
            kv: _KiralyVectors,
            men_to_indices: Dict[int, List[int]],
            similarities: np.array) -> np.array:


        key_women_candidates = self.key_women_candidates
        while free_men:
            man = free_men.popleft()
            indices = men_to_indices.get(man, [])
            if not indices:
                continue

            indices = np.asarray(indices)
            # active_mask = men_active[indices]

            if not kv.has_active(indices):
                woman = -1
            else:
                index = kv.get_first_active_index(indices)
                woman = key_women_candidates[index]

            if woman == -1:
                kv.update_man(man, indices)
            else:
                man_score = similarities[index]
                append_to_free_men  = kv.update_current_matches(man_score, man, woman, index)
                if append_to_free_men != -1:
                    free_men.append(append_to_free_men)

        return kv.current_matches

    def _create_clusters(self, current_matches: np.array, start_time: float) -> list:
        edge_list = [(m, w) for m,w in enumerate(current_matches) if w!=-1]

        edges_df = cudf.DataFrame(
            edge_list,
            columns=["src", "dst"],
            dtype="int32"
        )
        cu_graph = cugraph.Graph(directed=False)

        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        rmm.reinitialize(pool_allocator=False)

        cu_graph.from_cudf_edgelist(
            edges_df,
            source="src",
            destination="dst",
            renumber=True
        )

        components_df = cugraph.connected_components(cu_graph).to_pandas()
        clusters = components_df.groupby('labels')['vertex'].apply(set).tolist()
        self.execution_time = time() - start_time

        return clusters

    def process(self,
        graph: Tuple[np.array, np.array],
        encoded_data: BloomEncodedData,
        similarity_threshold: float = 0.1) -> list:
        """
        Docstring for process

        :param self: Description
        :param graph: Description
        :type graph: Graph
        :param encoded_data: Description
        :type encoded_data: BloomEncodedData
        :param similarity_threshold: Description
        :type similarity_threshold: float
        :return: Description
        :rtype: list
        """
        start_time = time()

        self.encoded_data = encoded_data
        self.similarity_threshold : float = similarity_threshold


        sorted_edges, similarities = self. \
            _get_sorted_edges_and_similarities(
                self._get_valid_edges_and_weights(graph))

        if sorted_edges.shape[0] == 0:
            return []

        men = cp.unique(sorted_edges[:,0])

        key_men_candidates = sorted_edges[:, 0]
        # men_candidates = sorted_edges

        # key_women_candidates = sorted_edges[:, 1]
        self.key_women_candidates = sorted_edges[:, 1]
        men_to_indices = defaultdict(list)

        for i, m in enumerate(key_men_candidates):
            men_to_indices[m].append(i)

        kv = _KiralyVectors(sorted_edges.shape[0], self.encoded_data.bounds[0],
                            self.encoded_data.bounds[1] - self.encoded_data.bounds[0])

        free_men = deque(men.tolist())
        current_matches = self._get_current_matches(free_men, kv,
                                        men_to_indices, similarities)

        return self._create_clusters(current_matches, start_time)
