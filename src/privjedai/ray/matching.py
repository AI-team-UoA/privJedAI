"""Matching methods of pprl"""
from __future__ import annotations
import time
from itertools import combinations, product
from typing import Literal, List, Dict, overload, Tuple, Union, Set
from dataclasses import dataclass
import logging
import numpy as np
from ordered_set import OrderedSet
from tqdm.auto import tqdm
import pandas as pd
import numexpr as ne
import ray
from privjedai.utils import _dice, _scm, _cosine, _jaccard
from privjedai.datamodel import  Block, EncodedData
from privjedai.encoded_data import BloomEncodedData, HomomorphicEcnodedData
from privjedai.evaluation import Evaluation
from privjedai.datamodel import PPRLFeature

# NOTE: OPENFHE
# import pprl.openfhe_similarity
# from openfhe import CCParamsBGVRNS, SecurityLevel, \
#     CryptoContext, GenCryptoContext, KeyPair, PKESchemeFeature # pylint: disable=no-name-in-module


@dataclass
class MatcherConfig:
    """Matcher config class"""
    batch_size : int
    threshold : float
    metric : str
    attributes : List[str]
    candidates : np.array
    unique_ids : np.array
    workers : int = 1

class Matcher(PPRLFeature):
    """Matcher class for pprl"""

    _method_name = 'Matcher'
    _method_short_name = 'Matcher'

    def __init__(self,
        batch_size : int = 10_000,
        threshold : float = 0.6,
        workers : int = 1,
        metric : Literal["dice", "scm", 'jaccard', "cosine"] = "dice",
        attributes : List[str] = None,
    ) -> None:
        super().__init__()
        self.matcher_config = MatcherConfig(batch_size=batch_size,
            threshold=threshold, metric=metric,
            attributes=attributes, candidates=None,
            unique_ids=None, workers=workers)
        self.execution_time : float = 0.0
        self.encoded_data : EncodedData
        self.ground_truth : pd.DataFrame
        if not ray.is_initialized():
            logging.getLogger("ray").setLevel(logging.ERROR)
            ray.init(log_to_driver=False, logging_level=logging.ERROR)


        # NOTE: OPENFHE
        # self.cc : CryptoContext
        # self.key_pair : KeyPair


    def update_graph(self, graph: tuple, threshold: float) -> tuple:
        """Updates graph based on a new threshold"""
        self.matcher_config.threshold = threshold
        edges, weights = graph

        mask = ne.evaluate("threshold < weights")

        # mask = self.matcher_config.threshold < weights
        edges = edges[mask]
        weights = weights[mask]
        return edges, weights

    def _candidates(self,
            extract_candidates: Union[Dict[str, Block], EncodedData]) -> None:


        _, first_value = next(iter(extract_candidates.items()))
        if isinstance(first_value, Block):
            candidates = set()
            for _, block in extract_candidates.items():
                for d1, d2 in combinations(block.entities.values(), 2):
                    for eid1, eid2 in product(d1, d2):
                        candidates.add((eid1,eid2))
            self.matcher_config.candidates = np.array(list(candidates))
        else:
            entity_ids_1 = list(extract_candidates.keys())
            entity_ids_2 = [list(v) for v in extract_candidates.values()]
            entity_ids_1_array = np.repeat(entity_ids_1, [len(v) for v in entity_ids_2])
            entity_ids_2_array = np.concatenate(entity_ids_2)
            self.matcher_config.candidates = np.column_stack([entity_ids_1_array, entity_ids_2_array])

        self.matcher_config.unique_ids = np.unique(self.matcher_config.candidates)

    @overload
    def predict(self,
            encoded_data: HomomorphicEcnodedData,
            blocks : Dict[str, Block] = None,
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs') -> tuple:
        """
        Similarity Computaation for Homomorphic Encrypted Data

        Args:

            ecnoded_data (HomomorphicEncodedData) : Homomorphic encoded data
            blocks (Dict[str, Block], optional): Dictionary of Blocks
            type_of_match (**'overlap','extension'**, optional): Type of match
            homomorphism (**'bgvs'**, optional): Homomorphism type

        Returns:
            Graph: The matching pairs

        """


    @overload
    def predict(self,
                encoded_data : BloomEncodedData,
                blocks: Dict[str, Block] = None)  -> tuple:
        """
        Vectorized  similarity computation for Bloom filters.

        Args:

            BloomEncodedData (ecnoded_data): Bloom Filters
            blocks (dict, optional): Dictionary of Blocks

        Returns:
            Graph: The matching pairs
        """



    def predict(self,
                encoded_data: EncodedData,
                blocks: Dict[str, Block] = None,
                # **kwargs
                ) -> tuple:
        """
        Vectorized  similarity computation for Bloom filters or HE.

        Args:

            BloomEncodedData (ecnoded_data): Bloom Filters
            blocks (dict, optional): Dictionary of Blocks
            **kwargs : Choose between he or bf

        Returns:
            Graph: The matching pairs
        """

        start_time = time.time()
        self.encoded_data = encoded_data

        if not encoded_data.skip_ground_truth:
            self.ground_truth = encoded_data.ground_truth

        if blocks:
            self._candidates(blocks)


        if isinstance(encoded_data, BloomEncodedData):
            self._method_info = """Matching with Bloom Filters"""
            graph = self._predict_from_bloom(encoded_data)
        elif isinstance(encoded_data, HomomorphicEcnodedData):
            raise NotImplementedError("Homomorphic matching is not implemented yet.")
            # self._method_info = """Matching with Homomorhpic Encoded Data """
            # graph = self._predict_from_vector(encoded_data, **kwargs)
        else:
            raise TypeError(f"encoded_data must be of type: BloomEncodedData \
                or HomomorphicEcnodedData, but its {type(encoded_data)}")

        self.execution_time = time.time() - start_time
        return graph

    # NOTE: OPENFHE
    # def _bgvs(self) -> None:
    #     parameters = CCParamsBGVRNS()
    #     sigma = 3.2                       # palisade: sigma
    #     security_level = SecurityLevel.HEStd_128_classic

    #     parameters.SetStandardDeviation(sigma)   # set sigma
    #     parameters.SetSecurityLevel(security_level)
    #     parameters.SetPlaintextModulus(65537)  # plaintext_modulus
    #     parameters.SetMultiplicativeDepth(1)

    #     self.cc : CryptoContext = GenCryptoContext(parameters)
    #     self.cc.Enable(PKESchemeFeature.PKE)  # palisade encryption
    #     self.cc.Enable(PKESchemeFeature.LEVELEDSHE) #palisade she
    #     self.cc.Enable(PKESchemeFeature.ADVANCEDSHE) #palisade she
    #     # Generate a public/private key pair
    #                                         # Palisade code
    #     self.key_pair : KeyPair = self.cc.KeyGen()  # LPKeyPair<DCRTPoly> keyPair;
    #                                         # keyPair = cc->KeyGen();

    #     # Generate the relinearization key
    #     self.cc.EvalSumKeyGen(self.key_pair.secretKey)
    # # cc->EvalSumKeyGen(keyPair.secretKey);


    def _configuration(self):
        return { "batch_size": self.matcher_config.batch_size,
            "threshold": self.matcher_config.threshold,
            "metric": self.matcher_config.metric,
            "attributes": self.matcher_config.attributes }


    # NOTE: OPENFHE
    def _predict_from_vector(self, encoded_data: HomomorphicEcnodedData,
            type_of_match: Literal['overlap', 'extension'] = 'overlap',
            homomorphism : Literal['bgvs'] = 'bgvs' ) -> tuple:
        raise NotImplementedError("Homomorphic matching is not implemented yet.")
        # if homomorphism == 'bgvs':
        #     self._bgvs()

        # matching_function = f'_{type_of_match}_jaccard'
        # matching_function = getattr(src.pprl.openfhe_similarity, matching_function)

        # attributes = self.matcher_config.attributes if self.matcher_config.attributes \
        #     else encoded_data.metadata.attributes

        # graph : Graph = Graph()

        # homomorphic_dict = encoded_data.encoded_dict

        # for id1, id2 in self.matcher_config.candidates:
        #     similarity = 0.0
        #     encoded_1 = homomorphic_dict[id1]
        #     encoded_2 = homomorphic_dict[id2]
        #     for attr in attributes:
        #         similarity += matching_function(self.cc, self.key_pair,
        #                                        encoded_1[attr], encoded_2[attr])
        #     avg_similarity = similarity / len(attributes)
        #     if avg_similarity >= self.matcher_config.threshold:
        #         graph.add_edge(id1, id2, weight=avg_similarity)

        # return graph


    @staticmethod
    @ray.remote
    def _predict_batches(
            batch_pairs : np.array,
            bloom_matrix: np.array,
            unique_ids_arr: np.array,
            threshold: float,
            metric: str = 'dice',
            ) -> List[Tuple[int, int, float]]:

        metrics_dict = {
            "dice" : _dice,
            "scm" : _scm,
            "jaccard": _jaccard,
            "cosine" : _cosine
        }


        _metric = metrics_dict.get(metric,
                            metrics_dict['dice'])

        avg_similarities : np.array = _metric(
                bloom_matrix[batch_pairs[:,0]],
                bloom_matrix[batch_pairs[:,1]])

        mask = avg_similarities >= threshold

        if not np.any(mask):
            return [], [], []

        valid_pairs = batch_pairs[mask]
        valid_scores = avg_similarities[mask]

        src_ids = unique_ids_arr[valid_pairs[:, 0]]
        tgt_ids = unique_ids_arr[valid_pairs[:, 1]]

        return src_ids, tgt_ids, valid_scores

    def _predict_create_bloom_matrix(self, attributes: list,
                            encoded_data: BloomEncodedData) -> np.array:

        n_attrs = len(attributes)
        length = encoded_data.metadata.length

        if self.matcher_config.unique_ids is None:
            bloom_filter_matrix = np.zeros((self.encoded_data.bounds[1],
                                n_attrs, length), dtype=np.bool)

            for entity_id, fields in self.encoded_data.encoded_dict.items():
                for attr_idx, attr in enumerate(attributes):
                    if fields[attr]:
                        indices = np.array(fields[attr])
                        bloom_filter_matrix[entity_id,
                        attr_idx,
                        indices]  = True


            return bloom_filter_matrix

        n_ids = len(self.matcher_config.unique_ids)
        length = encoded_data.metadata.length

        bloom_filter_matrix = np.zeros((n_ids, n_attrs, length), dtype=bool)

        # Fill the bloom matrix
        for eid_idx, eid in enumerate(self.matcher_config.unique_ids):
            fields = encoded_data.encoded_dict[eid]
            for attr_idx, attr in enumerate(attributes):
                if fields[attr]:
                    bloom_filter_matrix[eid_idx, attr_idx,
                            np.array(fields[attr])] = True

        return bloom_filter_matrix

    def _get_unique_ids(self) -> tuple:

        sorter = np.argsort(self.matcher_config.unique_ids)

        idx_pairs = sorter[
            np.searchsorted(self.matcher_config.unique_ids,
                    self.matcher_config.candidates, sorter=sorter)
            ]
        unique_ids_arr = self.matcher_config.unique_ids
        n_candidates = idx_pairs.shape[0]

        return n_candidates, unique_ids_arr, idx_pairs

    def _get_unique_ids_dense(self) -> tuple:
        unique_ids_arr = np.arange(0, self.encoded_data.bounds[1])
        ids_1 = self.encoded_data.bounds[0]
        ids_2 = self.encoded_data.bounds[1] - ids_1
        n_candidates = ids_1 * ids_2
        return n_candidates, unique_ids_arr

    def _future_wait_workers(self,
                futures: list,
                edges_list: list,
                edges_weights_list: list) -> tuple:

        finished, futures = ray.wait(futures, num_returns=1)
        for done in finished:
            src, tgt, weights = ray.get(done)
            edges_list.append(np.column_stack((src, tgt)))
            edges_weights_list.append(weights)

        return futures, edges_list, edges_weights_list

    def _match_all(self, n_candidates: int,
        bloom_matrix: np.array, unique_ids_arr: np.array,
        idx_pairs: np.array = None) -> tuple:

        edges_list = []
        edges_weights_list = []
        futures = []
        for start in tqdm(range(0, n_candidates, self.matcher_config.batch_size),
                    desc="Predicting batches", position=1, leave=False):
            end = min(start + self.matcher_config.batch_size, n_candidates)
            batch_pairs=idx_pairs[start:end]
            futures.append(
                self._predict_batches.remote(batch_pairs,
                                    bloom_matrix,
                                    unique_ids_arr,
                                    self.matcher_config.threshold,
                                    self.matcher_config.metric)
            )
            if len(futures) >= self.matcher_config.workers:
                futures, edges_list, edges_weights_list = \
                    self._future_wait_workers(
                        futures, edges_list, edges_weights_list)
        while futures:
            futures, edges_list, edges_weights_list = \
                self._future_wait_workers(
                    futures, edges_list, edges_weights_list)

        return np.concatenate(edges_list, axis=0), np.concatenate(edges_weights_list, axis=0)

    def _match_all_is_dense(self, n_candidates: int,
        bloom_matrix: np.array, unique_ids_arr: np.array) -> tuple:

        futures = []
        edges_list = []
        edges_weights_list = []
        for start in tqdm(range(0, n_candidates, self.matcher_config.batch_size),
                    desc="Predicting batches", position=1, leave=False):
            end = min(start + self.matcher_config.batch_size, n_candidates)
            batch_indices = np.arange(start, end)
            rows, cols = np.divmod(batch_indices,
                            self.encoded_data.bounds[1] - self.encoded_data.bounds[0])
            cols += self.encoded_data.bounds[0]
            batch_pairs = np.column_stack((rows, cols))
            futures.append(
                self._predict_batches.remote(batch_pairs,
                        bloom_matrix,
                        unique_ids_arr,
                        self.matcher_config.threshold,
                        self.matcher_config.metric)
            )
            if len(futures) >= self.matcher_config.workers:
                futures, edges_list, edges_weights_list = \
                    self._future_wait_workers(
                        futures, edges_list, edges_weights_list)

        while futures:
            futures, edges_list, edges_weights_list = \
                self._future_wait_workers(
                    futures, edges_list, edges_weights_list)
        return np.concatenate(edges_list, axis=0), np.concatenate(edges_weights_list, axis=0)

    def _predict_from_bloom(self, encoded_data: BloomEncodedData) -> tuple:

        attributes = self.matcher_config.attributes if self.matcher_config.attributes \
            else encoded_data.metadata.attributes

        n_candidates = 0
        bloom_matrix = self._predict_create_bloom_matrix(attributes, encoded_data)


        if self.matcher_config.unique_ids is not None:
            n_candidates, unique_ids_arr, idx_pairs = self._get_unique_ids()
            return self._match_all(n_candidates, bloom_matrix, unique_ids_arr, idx_pairs)

        n_candidates, unique_ids_arr = self._get_unique_ids_dense()
        return self._match_all_is_dense(n_candidates, bloom_matrix, unique_ids_arr)


    def evaluate(self,
                prediction : np.array,
                export_to_df: bool = False,
                with_classification_report: bool = False,
                verbose: bool = True) -> any:

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Encoded Data object the ground-truth file has not been set.")
        edges, _ = prediction

        edges = edges.astype(np.int32)
        eval_obj = Evaluation(self.encoded_data)


        id1_arr = self.encoded_data.ground_truth.iloc[:, 0].values
        offset = self.encoded_data.bounds[0]

        id2_arr = (self.encoded_data.ground_truth.iloc[:, 1].values + offset)

        # if self.matcher_config.unique_ids:
        #     id_to_idx = {eid: idx \
        #         for idx, eid in enumerate(self.matcher_config.unique_ids)}
        #     replace_func = np.vectorize(lambda x: id_to_idx.get(x, x))
        #     id1_arr = replace_func(id1_arr)
        #     id2_arr = replace_func(id2_arr)

        gt_pairs = np.column_stack((id1_arr, id2_arr)).astype(np.int32)

        def as_void(arr):
            return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        matches = np.intersect1d(as_void(edges), as_void(gt_pairs))

        eval_obj.calculate_scores(true_positives=matches.shape[0],
                                  total_matching_pairs=edges.shape[0])
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

    def stats(self) -> None:
        """NOT IMPLEMENTED"""
