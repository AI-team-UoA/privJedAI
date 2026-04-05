"""PPRL Metablocking Methods"""
import sys
from queue import PriorityQueue
from itertools import chain
from collections import defaultdict
from math import log10, sqrt
from time import time
from abc import ABC, abstractmethod
from typing import Dict
from scipy.sparse import csr_matrix, diags, coo_matrix
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from privjedai.evaluation import Evaluation
from privjedai.datamodel import PPRLFeature
from privjedai.encoded_data import BloomEncodedData
from privjedai.utils import chi_square



class AbstractComparisonCleaning(PPRLFeature):
    """Abstract class for Block cleaning
    """

    m_sparse: csr_matrix
    entity_ids: np.array
    block_keys: np.array

    def __init__(self) -> None:
        super().__init__()
        self.encoded_data : BloomEncodedData
        self.entity_id_block_key_pairs : list
        self.candidate_pairs : dict

    def process(
            self,
            encoded_data: BloomEncodedData,
            adjacent_bits : int = 1
    ) -> dict:
        """Main method for comparison cleaning

        Args:
            encoded_data (BloomEncodedData): bloom filters created from previous steps of pprl

        Returns:
            dict: cleaned blocks
        """
        start_time = time()
        self.encoded_data = encoded_data

        self.candidate_pairs : Dict[int, set] = defaultdict(set)

        self.entity_ids, self.block_keys, = encoded_data.get_entity_id_block_key_pairs(
                                                adjacent_bits
                                            )


        entity_indices = self.entity_ids

        num_of_entities = len(self.encoded_data.encoded_dict)
        entity_indices = self.entity_ids

        _, block_indices = np.unique(self.block_keys,
                            return_inverse=True)

        num_blocks = block_indices.max() + 1 \
            if len(block_indices) > 0 else 0


        data = np.ones(entity_indices.shape[0], dtype=bool)
        self.m_sparse = csr_matrix(
            (data, (entity_indices, block_indices)),
            shape=(num_of_entities, num_blocks),
            dtype=bool
        )


        self.candidate_pairs = self._apply_main_processing()

        self.execution_time = time() - start_time

        return self.candidate_pairs

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        _conf = self._configuration()
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join([f'\t{k}: {v}\n' for k, v in _conf.items()])
            if _conf.items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in self.encoded_data.attributes) +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )

    @abstractmethod
    def _apply_main_processing(self) -> dict:
        pass

    @abstractmethod
    def _configuration(self) -> dict:
        pass

    def evaluate(self,
                 prediction : dict,
                 export_to_df: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> any:
        """Function to evaluate meta-blocking methods f1-score, recall and precision

        Args:
            prediction (dict) : Blocks predicted from the blocking method
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """

        eval_obj = Evaluation(self.encoded_data)
        eval_obj.evaluate_candidate_pairs(prediction)
        return eval_obj.report(self.method_configuration(),
                               export_to_df,
                               with_classification_report,
                               verbose)

    def export_to_df(self, prediction: dict) -> pd.DataFrame:
        """creates a dataframe with the predicted pairs

        Args:
            prediction (dict): Predicted candidate pairs

        Returns:
            pd.DataFrame: Dataframe with the predicted pairs
        """
        pairs_list = []
        for id1, candidates in prediction.items():
            for candiadate_id in candidates:
                id2 = candiadate_id - self.encoded_data.bounds[0]
                pairs_list.append({'id1': id1, 'id2': id2})

        pairs_df = pd.DataFrame(pairs_list) if len(pairs_list) > 0 else pd.DataFrame(
                columns=['id1', 'id2'])

        return pairs_df


class AbstractMetablocking(AbstractComparisonCleaning, ABC):
    """Restructure a redundancy-positive block collection into a new
        one that contains substantially lower number of redundant
        and superfluous comparisons, while maintaining the original number of matching ones
    """


    def __init__(self) -> None:
        super().__init__()
        self._flags: np.array
        self._counters: np.array
        self._comparisons_per_entity: np.array
        self._distinct_comparisons: int
        self._neighbors: set = set()
        self.weighting_scheme: str

    def _apply_main_processing(self) -> dict:
        self._counters = np.empty([self.encoded_data.bounds[1]], dtype=float)
        self._flags = np.empty([self.encoded_data.bounds[1]], dtype=int)
        self._set_statistics()
        self._set_threshold()

        return self._prune_edges()

    def _get_sim_matrix(self) -> csr_matrix:
        _block_cardinalities = self._get_block_cardinalities()
        processed_m1 = self.m_sparse[:
            self.encoded_data.bounds[0]].multiply(_block_cardinalities)
        sim_matrix = processed_m1.dot(self.m_sparse[self.encoded_data.bounds[0]:].T)
        return sim_matrix


    def _snc(self, _counters):
        _limit = self.encoded_data.bounds[0]
        comp_e1 = self._comparisons_per_entity[:_limit]
        comp_e2 = self._comparisons_per_entity[_limit:]
        sqrt_comp_e1 = np.sqrt(comp_e1)
        sqrt_comp_e2 = np.sqrt(comp_e2)
        inv_sqrt_comp_e1 = np.divide(1.0, sqrt_comp_e1, out=np.zeros_like(sqrt_comp_e1), where=sqrt_comp_e1!=0)
        inv_sqrt_comp_e2 = np.divide(1.0, sqrt_comp_e2, out=np.zeros_like(sqrt_comp_e2), where=sqrt_comp_e2!=0)

        D1 = diags(inv_sqrt_comp_e1)
        D2 = diags(inv_sqrt_comp_e2)

        return D1 @ _counters @ D2

    def _snd(self, _counters):
        _limit = self.encoded_data.bounds[0]
        _counters_coo = _counters.tocoo()
        rows = _counters_coo.row
        cols = _counters_coo.col
        data = _counters_coo.data

        denominators = self._comparisons_per_entity[rows] + self._comparisons_per_entity[cols + _limit]
        new_data = np.zeros_like(data, dtype=float)
        valid_mask = denominators != 0
        new_data[valid_mask] = 2 * data[valid_mask] / denominators[valid_mask]

        return coo_matrix(
            (new_data, (rows, cols)),
            shape=_counters.shape).tocsr()

    def _snj(self, _counters):
        _limit = self.encoded_data.bounds[0]
        _counters_coo = _counters.tocoo()
        rows = _counters_coo.row
        cols = _counters_coo.col
        data = _counters_coo.data

        denominators = self._comparisons_per_entity[rows] + \
            self._comparisons_per_entity[cols + _limit] - data
        new_data = np.zeros_like(data, dtype=float)
        valid_mask = denominators != 0
        new_data[valid_mask] = data[valid_mask] / denominators[valid_mask]

        return coo_matrix(
            (new_data, (rows, cols)),
            shape=_counters.shape).tocsr()

    def _cosine(self, _counters):
        _limit = self.encoded_data.bounds[0]
        m1_sparse = self.m_sparse[:_limit]
        m2_sparse = self.m_sparse[_limit:]

        len_m1_sparse = m1_sparse.getnnz(axis=1).astype(float)
        len_m2_sparse = m2_sparse.getnnz(axis=1).astype(float)
        sqrt_len_m1_sparse = np.sqrt(len_m1_sparse)
        sqrt_len_m2_sparse = np.sqrt(len_m2_sparse)
        inv_sqrt_len_m1_sparse = np.divide(1.0,
                sqrt_len_m1_sparse,
                out=np.zeros_like(sqrt_len_m1_sparse),
                where=sqrt_len_m1_sparse!=0)
        inv_sqrt_len_m2_sparse = np.divide(1.0,
                sqrt_len_m2_sparse,
                out=np.zeros_like(sqrt_len_m2_sparse),
                where=sqrt_len_m2_sparse!=0)
        D1 = diags(inv_sqrt_len_m1_sparse)
        D2 = diags(inv_sqrt_len_m2_sparse)
        return D1 @ _counters @ D2

    def _dice(self, _counters):
        _limit = self.encoded_data.bounds[0]
        m1_sparse = self.m_sparse[:_limit]
        m2_sparse = self.m_sparse[_limit:]
        _counters_coo = _counters.tocoo()

        rows = _counters_coo.row
        cols = _counters_coo.col
        data = _counters_coo.data
        len_m1_sparse = m1_sparse.getnnz(axis=1).astype(float)
        len_m2_sparse = m2_sparse.getnnz(axis=1).astype(float)

        denominators = len_m1_sparse[rows] + \
            len_m2_sparse[cols]
        new_data = np.zeros_like(data, dtype=float)
        valid_mask = denominators != 0
        new_data[valid_mask] = 2 * data[valid_mask] / denominators[valid_mask]
        return coo_matrix(
            (new_data, (rows, cols)),
            shape=_counters.shape).tocsr()

    def _ecbs(self, _counters):
        _limit = self.encoded_data.bounds[0]
        m1_sparse = self.m_sparse[:_limit]
        m2_sparse = self.m_sparse[_limit:]

        len_m1_sparse = m1_sparse.getnnz(axis=1).astype(float)
        len_m2_sparse = m2_sparse.getnnz(axis=1).astype(float)

        num_of_bits = self.m_sparse.shape[1]
        inv_len_m1_sparse = np.divide(num_of_bits,
                    len_m1_sparse,
                    out=np.zeros_like(len_m1_sparse, dtype=float),
                    where=len_m1_sparse!=0)
        inv_len_m2_sparse = np.divide(num_of_bits,
                            len_m2_sparse,
                            out=np.zeros_like(len_m2_sparse, dtype=float),
                            where=len_m2_sparse!=0)
        log_inv_len_m1_sparse = np.log10(inv_len_m1_sparse,
                            out=np.zeros_like(inv_len_m1_sparse, dtype=float),
                            where=inv_len_m1_sparse!=0)
        log_inv_len_m2_sparse = np.log10(inv_len_m2_sparse,
                        out=np.zeros_like(inv_len_m2_sparse, dtype=float),
                        where=inv_len_m2_sparse!=0)

        D1 = diags(log_inv_len_m1_sparse)
        D2 = diags(log_inv_len_m2_sparse)
        return D1 @ _counters @ D2

    def _js(self, _counters):
        _limit = self.encoded_data.bounds[0]
        m1_sparse = self.m_sparse[:_limit]
        m2_sparse = self.m_sparse[_limit:]

        len_m1_sparse = m1_sparse.getnnz(axis=1).astype(float)
        len_m2_sparse = m2_sparse.getnnz(axis=1).astype(float)

        _counters_coo = _counters.tocoo()
        rows = _counters_coo.row
        cols = _counters_coo.col
        data = _counters_coo.data

        denominators = len_m1_sparse[rows] + \
            len_m2_sparse[cols] - data

        new_data = np.zeros_like(data, dtype=float)
        valid_mask = denominators != 0
        new_data[valid_mask] = data[valid_mask] / denominators[valid_mask]
        return coo_matrix(
            (new_data, (rows, cols)),
            shape=_counters.shape).tocsr()

    def _ejs(self, _counters):
        propability = self._js(_counters)
        _limit = self.encoded_data.bounds[0]
        inv_comparisons_e1 = np.divide(self._distinct_comparisons,
                            self._comparisons_per_entity[:_limit],
                            out=np.zeros_like(self._comparisons_per_entity[:_limit], dtype=float),
                            where=self._comparisons_per_entity[:_limit]!=0)
        inv_comparisons_e2 = np.divide(self._distinct_comparisons,
                            self._comparisons_per_entity[_limit:],
                            out=np.zeros_like(self._comparisons_per_entity[_limit:], dtype=float),
                            where=self._comparisons_per_entity[_limit:]!=0)
        log_inv_comparisons_e1 = np.log10(inv_comparisons_e1,
                            out=np.zeros_like(inv_comparisons_e1, dtype=float),
                            where=inv_comparisons_e1!=0)
        log_inv_comparisons_e2 = np.log10(inv_comparisons_e2,
                            out=np.zeros_like(inv_comparisons_e2, dtype=float),
                            where=inv_comparisons_e2!=0)

        D1 = diags(log_inv_comparisons_e1)
        D2 = diags(log_inv_comparisons_e2)
        return D1 @ propability @ D2

    def _x2(self, _counters):
        _limit = self.encoded_data.bounds[0]
        m1_sparse = self.m_sparse[:_limit]
        m2_sparse = self.m_sparse[_limit:]

        len_m1_sparse = m1_sparse.getnnz(axis=1).astype(float)
        len_m2_sparse = m2_sparse.getnnz(axis=1).astype(float)

        num_of_blocks = self.m_sparse.shape[1]

        _counters_coo = _counters.tocoo()
        rows = _counters_coo.row
        cols = _counters_coo.col
        data = _counters_coo.data

        R_i = len_m1_sparse[rows].astype(float)
        C_j = len_m2_sparse[cols].astype(float)

        numerator = num_of_blocks * (data * num_of_blocks - R_i * C_j) ** 2
        denominator = R_i * C_j * (num_of_blocks - R_i) * (num_of_blocks - C_j)
        new_data = np.zeros_like(data, dtype=float)
        valid_mask = denominator != 0
        new_data[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        return coo_matrix(
            (new_data, (rows, cols)),
            shape=_counters.shape).tocsr()

    def _get_weight(self, _counters) -> np.array:
        weighting_dict = {
            'CN-CBS': lambda : _counters,
            'CBS': lambda : _counters,
            'SN-CBS': lambda : _counters,
            'SNC': lambda : self._snc(_counters),
            'CNC': lambda : self._snc(_counters),
            'SND' : lambda : self._snd(_counters),
            'CND' : lambda : self._snd(_counters),
            'CNJ' : lambda : self._snj(_counters),
            'SNJ' : lambda : self._snj(_counters),
            'COSINE': lambda :  self._cosine(_counters),
            'DICE': lambda : self._dice(_counters),
            'ECBS': lambda : self._ecbs(_counters),
            'JS': lambda : self._js(_counters),
            'EJS': lambda : self._ejs(_counters),
            'X2': lambda : self._x2(_counters)
        }
        if self.weighting_scheme not in weighting_dict:
            raise ValueError("This weighting scheme does not exist")

        return weighting_dict[self.weighting_scheme]()

    def _process_entities(self) -> np.array:

        _limit = self.encoded_data.bounds[0]

        m1_sparse = self.m_sparse[:_limit].astype(float)
        m2_sparse = self.m_sparse[_limit:].astype(float)

        if self.weighting_scheme in {'CN-CBS', 'CNC', 'CND', 'CNJ'}:
            s1 = m1_sparse.sum(axis=0).A1
            s2 = m2_sparse.sum(axis=0).A1
            cardinality = s1 * s2
            weights = np.zeros_like(cardinality, dtype=float)
            non_zero_mask = cardinality > 0
            weights[non_zero_mask] = 1.0 / cardinality[non_zero_mask]
            w = diags(weights)

            return m1_sparse @ w @ m2_sparse.T

        if self.weighting_scheme in {'SN-CBS', 'SNC', 'SND', 'SNJ'}:
            s1 = m1_sparse.sum(axis=0).A1
            s2 = m2_sparse.sum(axis=0).A1
            block_size = s1 + s2
            weights = np.zeros_like(block_size, dtype=float)
            non_zero_mask = block_size > 0
            weights[non_zero_mask] = 1.0 / block_size[non_zero_mask]
            w = diags(weights)

            return m1_sparse @ w @ m2_sparse.T

        return m1_sparse @ m2_sparse.T


    # def _get_vectorized_weights(self, c) -> np.array:
    #     _limit = self.encoded_data.bounds[0]

    #     c1 = c2 = None
    #     if self.weighting_scheme in ['SNC' , 'CNC', 'CND', 'SND',
    #                         'CNJ', 'SNJ', 'EJS']:
    #         comp_1 = self._comparisons_per_entity[:_limit]
    #         comp_2 = self._comparisons_per_entity[_limit:]
    #         c1 = comp_1[c.row]
    #         c2 = comp_2[c.col]


    #     return self._get_weight(c, c1, c2)

    def _set_statistics(self) -> None:
        if self.weighting_scheme not in {'EJS',
                'CNC', 'SNC', 'SND', 'CND', 'CNJ', 'SNJ'} :
            return
        m1_sparse = self.m_sparse[:self.encoded_data.bounds[0]]
        m2_sparse = self.m_sparse[self.encoded_data.bounds[0]:]
        c = m1_sparse.dot(m2_sparse.T)
        comp_e1 = c.getnnz(axis=1)
        comp_e2 = c.getnnz(axis=0)
        self._comparisons_per_entity = np.concatenate([comp_e1, comp_e2]).astype(float)
        self._distinct_comparisons = float(c.nnz) / 2



    @abstractmethod
    def _set_threshold(self):
        pass

    @abstractmethod
    def _prune_edges(self) -> dict:
        pass


class WeightedEdgePruning(AbstractMetablocking):
    """A Meta-blocking method that retains all comparisons \
        that have a weight higher than the average edge weight in the blocking graph.
    """

    _method_name = "Weighted Edge Pruning"
    _method_short_name: str = "WEP"
    _method_info = "A Meta-blocking method that retains all comparisons " + \
                "that have a weight higher than the average edge weight in the blocking graph."

    _sim_csr : csr_matrix = None

    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__()
        self.weighting_scheme = weighting_scheme
        self._node_centric = False
        self._num_of_edges: float
        self._threshold: float
        self._retained_neighbors: set = set()

    def _prune_edges(self) -> dict:

        _limit = self.encoded_data.bounds[0] if not self._node_centric \
            else len(self.encoded_data.encoded_dict)
        invalid_mask = self._threshold > self._sim_csr.data

        self._sim_csr.data[invalid_mask] = 0.0
        self._sim_csr.eliminate_zeros()

        indptr = self._sim_csr.indptr
        indices = self._sim_csr.indices
        for entity_id in range(_limit):
            start_idx = indptr[entity_id]
            end_idx = indptr[entity_id + 1]
            if start_idx < end_idx:
                valid_neighbors = indices[start_idx:end_idx] + _limit
                self.candidate_pairs[entity_id].update(valid_neighbors)

        return self.candidate_pairs

    def _process_entity(self, entity_id: int) -> None:
        pass



    def _set_threshold(self):
        self._num_of_edges = 0.0
        self._threshold = 0.0


        _counters = self._process_entities()
        self._num_of_edges = _counters.nnz

        _weights = self._get_weight(_counters)

        self._threshold = _weights.sum() / self._num_of_edges
        self._sim_csr = _weights


    def _configuration(self) -> dict:
        return {
            "Node centric" : self._node_centric,
            "Weighting scheme" : self.weighting_scheme
        }


class CardinalityEdgePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains the comparisons \
        that correspond to the top-K weighted edges in the bit graph.
    """

    _method_name = "Cardinality Edge Pruning"
    _method_short_name: str = "CEP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                        "that correspond to the top-K weighted edges in the bit graph."

    def __init__(self, weighting_scheme: str = 'JS') -> None:
        super().__init__(weighting_scheme)

    def _prune_edges(self) -> dict:

        _counters = self._process_entities()
        _weights = self._get_weight(_counters)

        k = self._threshold
        if k == 0:
            return self.candidate_pairs

        k = min(self._threshold, _weights.nnz)
        if k == _weights.nnz:
            top_k_1d_indices = np.arange(_weights.nnz)
        else:
            top_k_1d_indices = np.argpartition(_weights.data, -k)[-k:]

        row_indices = np.repeat(
            np.arange(_weights.shape[0]),
            np.diff(_weights.indptr)
        )

        non_zero_mask = np.nonzero(_weights.data[top_k_1d_indices])[0]

        # Apply the mask to keep only the valid indices
        top_k_1d_indices = top_k_1d_indices[non_zero_mask]


        for entity_id, neighbor_id in zip(row_indices[top_k_1d_indices],
                            _weights.indices[top_k_1d_indices]):
            self.candidate_pairs[entity_id].add(neighbor_id + self.encoded_data.bounds[0])

        return self.candidate_pairs

    def _set_threshold(self) -> None:
        self._threshold = self.block_keys.shape[0] // 2


class CardinalityNodePruning(CardinalityEdgePruning):
    """A Meta-blocking method that retains for every entity, \
        the comparisons that correspond to its top-k weighted edges in the blocking graph."
    """

    _method_name = "Cardinality Node Pruning"
    _method_short_name: str = "CNP"
    _method_info = "A Meta-blocking method that retains for every entity, " + \
                    "the comparisons that correspond to its top-k" + \
                    " weighted edges in the blocking graph."
    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._nearest_entities: dict
        self._node_centric = True
        self._top_k_edges: PriorityQueue

    def _prune_edges(self) -> dict:

        _counters = self._process_entities()
        _weights = self._get_weight(_counters)

        self.candidate_pairs = defaultdict(set)

        for i in range(_weights.shape[0]):
            start_idx = _weights.indptr[i]
            end_idx = _weights.indptr[i + 1]
            row_len = end_idx - start_idx

            if row_len == 0:
                continue

            row_data = _weights.data[start_idx:end_idx]
            row_cols = _weights.indices[start_idx:end_idx]

            actual_k = min(self._threshold, row_len)
            if row_len > actual_k:
                part_idx = np.argpartition(row_data, -actual_k)[-actual_k:]
                non_zero_mask = np.nonzero(row_data[part_idx])[0]
                part_idx = part_idx[non_zero_mask]
            else:
                part_idx = np.nonzero(row_data)[0]


        # Apply the mask to keep only the valid indices



            self.candidate_pairs[i].update(row_cols[part_idx] + self.encoded_data.bounds[0])

        _weights_col = _weights.tocsc()
        for j in range(_weights_col.shape[1]):
            start_idx = _weights_col.indptr[j]
            end_idx = _weights_col.indptr[j + 1]
            col_len = end_idx - start_idx

            if col_len == 0:
                continue

            col_data = _weights_col.data[start_idx:end_idx]
            col_rows = _weights_col.indices[start_idx:end_idx]

            actual_k = min(self._threshold, col_len)
            if col_len > actual_k:
                part_idx = np.argpartition(col_data, -actual_k)[-actual_k:]
                non_zero_mask = np.nonzero(col_data[part_idx])[0]
                part_idx = part_idx[non_zero_mask]
            else:
                part_idx = np.nonzero(col_data)[0]

            for e_id in col_rows[part_idx]:
                self.candidate_pairs[int(e_id)].add(j + self.encoded_data.bounds[0])

        return self.candidate_pairs


    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return True
        if entity_id in self._nearest_entities[neighbor_id]:
            return entity_id < neighbor_id
        return True

    def _set_threshold(self) -> None:
        num_of_entities = len(self.encoded_data.encoded_dict)
        bit_assignments = self.block_keys.shape[0]
        self._threshold = max(1, bit_assignments // num_of_entities)



class ReciprocalCardinalityNodePruning(CardinalityNodePruning):
    """A Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are among the top-k weighted  \
        ones for both adjacent entities/nodes.
    """

    _method_name = "Reciprocal Cardinality Node Pruning"
    _method_short_name: str = "RCNP"
    _method_info = "A Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are among " + \
                    "the top-k weighted ones for both adjacent entities/nodes."

    def __init__(self, weighting_scheme: str = 'CN-CBS') -> None:
        super().__init__(weighting_scheme)

    def _is_valid_comparison(self, entity_id: int, neighbor_id: int) -> bool:
        if neighbor_id not in self._nearest_entities:
            return False
        if entity_id in self._nearest_entities[neighbor_id]:
            return  entity_id < neighbor_id
        return False

class WeightedNodePruning(WeightedEdgePruning):
    """A Meta-blocking method that retains for every entity, the comparisons \
        that correspond to edges in the blocking graph that are exceed \
        the average edge weight in the respective node neighborhood.
    """

    _method_name = "Weighted Node Pruning"
    _method_short_name: str = "WNP"
    _method_info = "A Meta-blocking method that retains for every entity, the comparisons \
                    that correspond to edges in the blocking graph that are exceed \
                    the average edge weight in the respective node neighborhood."

    _weights : csr_matrix


    def __init__(self, weighting_scheme: str = 'CBS') -> None:
        super().__init__(weighting_scheme)
        self._average_weight: np.array
        self._node_centric = True



    def _prune_edges(self) -> dict:
        c = self._weights.tocoo()
        weights = c.data

        mask = (self._average_weight[c.row] <= weights) | \
            (self._average_weight[c.col + self.encoded_data.bounds[0]] <= weights)


        mask = mask & (c.row < c.col + self.encoded_data.bounds[0])
        mask = mask & (weights > 0)
        valid_rows = c.row[mask]
        valid_cols = c.col[mask] + self.encoded_data.bounds[0]

        for entity_id, neighbor_id in zip(valid_rows, valid_cols):
            self.candidate_pairs[entity_id].add(neighbor_id)
        return self.candidate_pairs


    def _set_threshold(self):

        _counters = self._process_entities()
        self._weights = self._get_weight(_counters)

        d1_sums = np.array(self._weights.sum(axis=1)).flatten()
        d1_counts = self._weights.getnnz(axis=1)
        valid_mask = d1_counts > 0
        d1_averages = np.zeros_like(d1_sums, dtype=float)
        d1_averages[valid_mask] = d1_sums[valid_mask] / d1_counts[valid_mask]

        d2_sums = np.array(self._weights.sum(axis=0)).flatten()
        d2_counts = self._weights.getnnz(axis=0)
        valid_mask = d2_counts > 0
        d2_averages = np.zeros_like(d2_sums, dtype=float)
        d2_averages[valid_mask] = d2_sums[valid_mask] / d2_counts[valid_mask]

        self._average_weight = np.concatenate([d1_averages, d2_averages])



    # def _verify_valid_entities(self, entity_id: int) -> None:
    #     if entity_id not in self._entity_index:
    #         return
    #     self._retained_neighbors.clear()
    #     for neighbor_id in self._valid_entities:
    #         _weight = self._get_valid_weight(entity_id, neighbor_id)
    #         if _weight:
    #             self._retained_neighbors.add(neighbor_id)
    #     if len(self._retained_neighbors) > 0:
    #         self.candidate_pairs[entity_id] = self._retained_neighbors.copy()

class BLAST(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons \
        that correspond to edges in the blocking graph that are exceed 1/4 of the sum \
        of the maximum edge weights in the two adjacent node neighborhoods.
    """

    _method_name = _method_short_name = "BLAST"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                "that correspond to edges in the blocking graph that are exceed 1/4 of the sum " + \
                "of the maximum edge weights in the two adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'X2') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        edge_threshold = (self._average_weight[entity_id] + self._average_weight[neighbor_id]) / 4
        return edge_threshold <= weight and entity_id < neighbor_id

    def _update_threshold(self, entity_id: int) -> None:
        if entity_id not in self._entity_index:
            return
        self._average_weight[entity_id] = 0.0
        for neighbor_id in self._valid_entities:
            self._average_weight[entity_id] = \
                max(self._average_weight[entity_id], self._get_weight(entity_id, neighbor_id))

class ReciprocalWeightedNodePruning(WeightedNodePruning):
    """Meta-blocking method that retains the comparisons\
        that correspond to edges in the blocking graph that are \
        exceed the average edge weight in both adjacent node neighborhoods.
    """

    _method_name = "Reciprocal Weighted Node Pruning"
    _method_short_name: str = "RWNP"
    _method_info = "Meta-blocking method that retains the comparisons " + \
                    "that correspond to edges in the blocking graph that are " + \
                    "exceed the average edge weight in both adjacent node neighborhoods."

    def __init__(self, weighting_scheme: str = 'CN-CBS') -> None:
        super().__init__(weighting_scheme)

    def _get_valid_weight(self, entity_id: int, neighbor_id: int) -> float:
        weight = self._get_weight(entity_id, neighbor_id)
        return weight if ((self._average_weight[entity_id] <= weight and \
                             self._average_weight[neighbor_id] <= weight) and
                                entity_id < neighbor_id) else 0
