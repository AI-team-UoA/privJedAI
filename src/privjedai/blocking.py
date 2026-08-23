"""PPRL Blocking Methods"""
import math
import random
from collections import defaultdict
from typing import List, Literal, Dict, Union, Any
import time
from abc import abstractmethod

import faiss
from bitarray import bitarray
import numpy as np

from privjedai.datamodel import Block, PPRLFeature, EncodedData
from privjedai.encoded_data import BloomEncodedData
from privjedai.evaluation import Evaluation



class AbstractBlockProcessing(PPRLFeature):
    """
    Abstract Class for blocking
    """

    def __init__(self):
        super().__init__()
        self.blocks: dict | None = None
        self.attributes: list = []
        self.encoded_data: EncodedData | None = None
        self.execution_time: float = 0.0
        self.blocks_with_keys : np.ndarray | None = None

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        if not self.encoded_data:
            raise AttributeError("Encoded data must be instantiated")
        _configuration = self._configuration()
        _attributes = self.attributes if self.attributes else []
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join([f'\t{k}: {v}\n' for k, v in _configuration.items()])
             if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(_attributes) +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )

    def evaluate(self,
                 prediction: dict,
                 export_to_df: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True) -> dict:
        """Function to evaluate meta-blocking methods f1-score, recall and precision

        Args:
            prediction (dict) : Blocks predicted from the blocking method
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            verbose (bool): Printing Evaluation

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """

        eval_obj = Evaluation(self.encoded_data)
        eval_obj.evaluate_candidate_pairs(prediction)
        return eval_obj.report(self.method_configuration(),
                               export_to_df,
                               with_classification_report,
                               verbose)
    def evaluate_blocks(self,
                        export_to_df : bool = False,
                        with_classification_report: bool = False,
                        verbose: bool = True) -> dict:
        """Evaluate the generated blocks using the evaluate method.

        Args:
            export_to_df (bool): Whether to export the evaluation results as a DataFrame.
            with_classification_report (bool): Whether to include a classification report in the evaluation output.
            verbose (bool): Whether to print detailed evaluation results to the console.

        Returns:
            dict: The evaluation results, which may include metrics such as F1-score, recall, precision, and optionally a classification report.
        """

        if self.blocks_with_keys is None:
            raise ValueError("Blocks have not been generated yet. Please run build_blocks() first.")

        eval_obj = Evaluation(self.encoded_data)
        eval_obj.evaluate_blocks(self.blocks_with_keys, self.encoded_data.bounds[0])
        return eval_obj.report(self.method_configuration(), export_to_df, with_classification_report, verbose)



    def _configuration(self) -> dict:  #pragma: no cover
        return {}

    @staticmethod
    def _clean_blocks(blocks: dict):
        new_blocks = {}
        for key, block in blocks.items():
            if len(block) != 0:
                new_blocks[key] = block
        return new_blocks

    @staticmethod
    def _clean_blocks_with_keys(blocks_with_keys: np.ndarray,
                                limit_: int) -> np.ndarray:


        mask_1 = blocks_with_keys[:, 1] < limit_
        mask_2 = blocks_with_keys[:, 1] >= limit_

        blocks_1 = np.unique(blocks_with_keys[mask_1, 0])
        blocks_2 = np.unique(blocks_with_keys[mask_2, 0])

        valid_blocks_id = np.intersect1d(blocks_1, blocks_2)
        final_mask = np.isin(blocks_with_keys[:, 0],
                             valid_blocks_id,
                             assume_unique=True)

        return blocks_with_keys[final_mask]







class AbstractBlockBuilding(AbstractBlockProcessing):
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str
    _method_short_name: str
    _index_time : float
    _blocking_time : float

    def __init__(self, seed: int = 42):
        super().__init__()
        self.original_num_of_blocks = 0
        self.seed: int = seed
        self.blocks: dict = {}


    def _get_record_keys(self, bloom_filters) -> List:
        concat_bits = bitarray()
        for attr in self.attributes:
            if attr in bloom_filters:
                concat_bits.extend(bloom_filters[attr])

        record_keys = self._block_record(concat_bits)
        return record_keys

    def _create_blocks(self) -> Dict[int, set]:
        blocks_d0 = defaultdict(list)
        blocks_d1 = defaultdict(list)

        for idx, bloom_filters in self.encoded_data.bitarray_dict.items():
            record_keys = self._get_record_keys(bloom_filters)
            target = blocks_d0 if idx < self.encoded_data.bounds[0] else blocks_d1
            for key in record_keys:
                target[key].append(idx)

        candidate_pairs = defaultdict(set)

        common_keys = blocks_d0.keys() & blocks_d1.keys()


        filtered_pairs = []
        for key in common_keys:
            d0_ids = blocks_d0[key]
            d1_ids = blocks_d1[key]
            for d0_id in d0_ids:
                candidate_pairs[d0_id].update(d1_ids)
            filtered_pairs.extend((key, idx) for idx in d0_ids)
            filtered_pairs.extend((key, idx) for idx in d1_ids)

        self.blocks_with_keys = np.array(filtered_pairs, dtype=np.int64)

        return candidate_pairs

    def build_blocks(
            self,
            encoded_data: BloomEncodedData,
            attributes: List[str] = None,
    ) -> dict:
        """Main method of Blocking in a dataset

            Args:
                encoded_data (BloomEncodedData) : Data bloom filters
                attributes (list, optional): Attributes columns of the datasets
                    that will be processed. Defaults to None. \
                    If not provided, all attributes are selected.
            Returns:
                Dict[token, Block]: Dictionary of blocks.
        """
        _start_time = time.time()
        self.encoded_data: BloomEncodedData = encoded_data
        bloom_attributes = encoded_data.metadata.attributes

        if attributes and bloom_attributes:
            if set(attributes) > set(bloom_attributes):
                raise ValueError(f"Attributes must be a subset "
                                 f"of the attributes in Bloom Filters : {bloom_attributes}")
            if attributes:
                self.attributes = attributes
        elif bloom_attributes:
            self.attributes = bloom_attributes
        else:
            raise ValueError("Must define attributes\n")


        self._fit()

        _end_time = time.time()

        self._index_time = _end_time - _start_time

        _start_time = time.time()




        blocks: Dict[int, set] = self._create_blocks()

        self.original_num_of_blocks = len(blocks)
        self.blocks = self._clean_blocks(blocks)

        self.execution_time = time.time() - _start_time + self._index_time
        self._blocking_time = self.execution_time - self._index_time
        if self.blocks_with_keys is not None:
            self.blocks_with_keys = self._clean_blocks_with_keys(self.blocks_with_keys, self.encoded_data.bounds[0])
            self.encoded_data.set_blocks_with_keys(self.blocks_with_keys)

        return self.blocks


    @abstractmethod
    def _fit(self) -> None:
        pass  # pragma: no cover

    def _block_record(self, bf: bitarray) -> List[str]:
        pass  # pragma: no cover

    def _configuration(self) -> dict:
        return {}  # pragma: no cover


class LSHBlocker(AbstractBlockBuilding):
    """
    Implements Locality-Sensitive Hashing (LSH)-based blocking for
    Bloom-filter–encoded records.

    Each encoded record produces Λ (lambda) blocking keys, each of
    length Ψ (psi) bits. These keys are used to group similar records
    together before performing expensive pairwise comparisons.

    Attributes:
        psi (int): Number of bit positions per key (Ψ).
        lambda_ (int): Number of blocking keys per record (Λ).
        prune_ratio (float): Ratio defining how frequently occurring
            or rare bit positions are pruned from the candidate pool.
        prune_sample (int): Number of records sampled to estimate the bit
            frequency distribution.
        seed (int): Random seed for reproducibility.
    """
    _method_name = "LSHBlocker"
    _method_info = "LSHBlocker"
    _method_short_name = "LSHBlocker"

    def __init__(
            self,
            psi: int = 36,
            lambda_: int = 3,
            prune_ratio: float = 0.6,
            prune_sample: int = 1000,
            seed: int = 42
    ):
        """
        Initialize LSHBlocker

        Args:
            psi (int): number of bit positions per key (Ψ)
            lambda_ (int): number of keys per record (Λ)
            prune_ratio (float): ratio of most frequent/uncommon bit positions to prune
            prune_sample (int): number of records to sample for frequency estimation
        """
        super().__init__(seed=seed)
        if psi < 1 or lambda_ < 1:
            raise ValueError(f"Both Values psi and lambda_ must be positive numbers : {psi}, {lambda_}")

        self._rng: random.Random
        self.psi = psi
        self.lambda_ = lambda_
        self.prune_ratio = prune_ratio
        self.prune_sample = prune_sample
        self._bit_positions = None  # will hold usable bit indices
        self._encoded_data: BloomEncodedData

    def _configuration(self):
        return {"psi": self.psi,
                "lambda_": self.lambda_,
                "prune_ratio": self.prune_ratio,
                "prune_sample": self.prune_sample,
                "seed": self.seed}

    def _select_bit_positions(self, bf_list: List[bitarray]) -> List[int]:
        """
        Determine a candidate pool of bit positions excluding too frequent or rare bits.
        Frequency estimated from initial sample.
        """
        m = len(bf_list[0])
        freq = np.zeros(m, dtype=int)
        sample = bf_list if len(bf_list) <= self.prune_sample else bf_list[: self.prune_sample]
        for bf in sample:
            freq += np.array(bf.tolist(), dtype=int)

        # normalize to frequency ratio
        freq_ratio = freq / len(sample)


        valid_mask = np.asarray(
            (freq_ratio <= self.prune_ratio) & (freq_ratio >= (1 - self.prune_ratio)),
            dtype=bool
        )
        candidates = np.nonzero(valid_mask)[0].tolist()
        if len(candidates) < self.psi:
            raise ValueError("Not enough bit positions after pruning")
        return candidates

    # Change fit to work with concatenated bloom_filters
    def _fit(self):
        """Initialize blocker by sampling to prune and selecting usable bit positions."""
        bitarray_dict: Dict[int, Dict[str, bitarray]] = self.encoded_data.bitarray_dict

        bitarray_list: List[bitarray] = [
            sum((array for attr, array in attr_bitarrays.items()
                 if attr in self.attributes), bitarray())
            for attr_bitarrays in bitarray_dict.values()
        ]

        self._rng = random.Random(self.seed)
        self._bit_positions = self._select_bit_positions(bitarray_list)

    def _block_record(self, bf: bitarray) -> List[str]:
        """
        Generate Λ blocking keys for a single Bloom filter record.
        Each key is Ψ bits sampled from bf at selected positions.
        """
        keys = []
        for _ in range(self.lambda_):
            positions = self._rng.sample(self._bit_positions, self.psi)
            bits = ''.join('1' if bf[j] else '0' for j in positions)
            # optionally, can hash this bit string to reduce size
            keys.append(bits)
        return keys


class BitBlocker(AbstractBlockBuilding):
    """
    LSH-based blocking for Bloom-filter-encoded records.

    Each record produces λ blocking keys of length ψ bits
    based on their Hamming values.

    Parameters
    ----------
    psi : int, default=36
        Number of bits per blocking key.
    lambda_ : int, default=3
        Number of blocking keys per record.
    seed : int, default=42
        Random seed for reproducibility.
    """

    _method_name = "BitBlocker"
    _method_info = "BitBlocker"
    _method_short_name = "BitBlocker"
    
    hash_indices_np: np.ndarray
    powers_of_2: np.ndarray

    @staticmethod
    def auto_psi_lambda(encoded_data: BloomEncodedData,
                        attributes : List = None,
                        threshold: float = 0.5,
                        delta: float = 0.1,
                        max_lambda_: int = 150
                        ) -> dict:
        """
        Estimate blocking parameters `psi` and `lambda_` for `BitBlocker`.

        This method computes parameter values used to generate blocking keys from
        Bloom-filter-encoded records. It searches for the largest feasible `psi`
        (number of sampled bits per key) such that the corresponding `lambda_`
        (number of hash tables / keys per record), derived from the target
        probability constraints, does not exceed `max_lambda_`.

        Args:
            encoded_data (BloomEncodedData):
                Encoded dataset containing Bloom filter metadata.
            attributes (List, optional):
                Subset of attributes to consider. If `None`, all available encoded
                attributes are used.
            threshold (float, optional):
                Percentage of Hamming Distance Threshold (max allowed differences).
                Defaults to `0.5`.
            delta (float, optional):
                Target upper bound for failure probability when deriving `lambda_`.
                Defaults to `0.1`.
            max_lambda_ (int, optional):
                Maximum allowed value for `lambda_`. The search stops once this bound
                is exceeded. Defaults to `150`.

        Returns:
            dict:
                Dictionary with selected parameters:
                - `'psi'` (int): selected number of sampled bit positions per key.
                - `'lambda_'` (int): selected number of blocking keys (hash tables).
        """
        bloom_size = encoded_data.metadata.length
        attr_len = len(attributes) if attributes else len(encoded_data.metadata.attributes)


        m = bloom_size * attr_len
        t = int(threshold * m)
        p = 1 - (t / m)

        best_psi = 1
        best_lambda_ = 1
        
        for k in range(1, m):

            probability_of_no_collision_in_one_table = 1 - (p**best_psi)
            if (probability_of_no_collision_in_one_table <= 0
                    or probability_of_no_collision_in_one_table >= 1):
                break
            lambda_ = math.ceil(math.log(delta) / math.log(probability_of_no_collision_in_one_table))
            if lambda_ > max_lambda_:
                break
            best_psi = k
            best_lambda_ = lambda_


        return {'psi' : best_psi, 'lambda_': best_lambda_}




    def __init__(self,
                 psi: int = 36,
                 lambda_: int = 3,
                 seed: int = 42):
        """
        Initializer of BitBlocker

        Args:
            psi (int): Number of bits per blocking key
            lambda_ (int): Number of blocking keys per record
            seed (int): Random seed for reproducibility

        """
        super().__init__(seed=seed)
        if psi < 1 or lambda_ < 1:
            raise ValueError(f"Both Values psi and lambda_ must be positive numbers : {psi}, {lambda_}")

        self.psi = psi
        self.lambda_ = lambda_
        self.hash_len: int
        self.encoded_data: BloomEncodedData
        self.rng: random.Random
        self.hash_indices: tuple

    def _fit(self) -> None:
        if self.encoded_data.metadata.length % 4 != 0:
            raise ValueError("Bloom Filters' length must multiple of 4.")

        self.hash_len = self.encoded_data.metadata.length * len(self.attributes)
        self.rng = random.Random(self.seed)
        self.hash_indices = tuple(self.rng.sample(range(self.hash_len), self.psi)
                                  for _ in range(self.lambda_))
        
        self.hash_indices_np = np.array(self.hash_indices)          # (n_tables, bits_per_table)
        self.powers_of_2 = (2 ** np.arange(self.hash_indices_np.shape[1])).astype(np.int64)



    def _block_record(self, bf: bitarray) -> List[int]:
        bf_np = np.frombuffer(bf.unpack(), dtype=np.uint8)  # shape: (n_bits,)

        n_tables = len(self.hash_indices)

        bits = bf_np[self.hash_indices_np]  # shape: (n_tables, bits_per_table)
        block_keys = np.dot(bits, self.powers_of_2)  # shape: (n_tables,)
        powers = self.powers_of_2
        table_blocks = bits @ powers  # shape: (n_tables,)
        block_keys = table_blocks * n_tables + np.arange(n_tables)  # shape:
        return block_keys



class FAISSBlocking(AbstractBlockBuilding):
    
    """
    A blocking implementation using FAISS for efficient similarity-based blocking.

    This class builds blocks by performing approximate nearest neighbor search
    on Bloom filter encodings of entities using FAISS.
    It creates blocks where each block contains entities that are similar to
    each other based on their binary vector representations.

    Attributes:
        top_k (int): Number of nearest neighbors to retrieve for each entity.
        index (faiss.IndexBinaryFlat): FAISS binary index for efficient similarity search.
        neighbors (np.array): Array containing neighbor indices from FAISS search.
        distances (np.array): Array containing distances to neighbors from FAISS search.
    """

    _method_name = "FAISS Blocking"
    _method_short_name: str = "FAISS"
    _method_info = "FAISS blocking."



    def __init__(self, index_type: Literal['flat', 'hnsw', 'multihash'] = 'flat'):
        super().__init__()
        self.top_k: int = 1
        self.encoded_data: BloomEncodedData
        self.neighbors : np.ndarray
        self.distances : np.ndarray
        self.index: faiss.IndexBinaryHNSW | faiss.IndexBinaryMultiHash | faiss.IndexBinaryFlat
        self.configuration : Dict[str, Any] = {'index_type' : index_type.lower()}

    def _init_vector(self, bitarray_dict: dict):
        bitarray_list: List[bitarray] = [
            sum((array for attr, array in attr_bitarrays.items()
                 if attr in self.attributes), bitarray())
            for attr_bitarrays in bitarray_dict.values()
        ]

        init_vector = np.ascontiguousarray([np.frombuffer(b.tobytes(), dtype=np.uint8)
                           for b in bitarray_list], dtype=np.uint8)

        return init_vector, bitarray_list


    def configure_hsnw(self, hnsw_m : int = 32):
        self.configuration['hnsw_m'] = hnsw_m

    def _set_index(self, vector_size: int):
        if 'hnsw' == self.configuration['index_type']:
            self.index = faiss.IndexBinaryHNSW(vector_size,
                                   self.configuration.get('hnsw_m', 32))
            # self.index.metric_type = faiss.METRIC_Jaccard
        elif 'multihash' == self.configuration['index_type']:
            lambda_: int = 8
            psi = vector_size // lambda_
            self.index = faiss.IndexBinaryMultiHash(vector_size, lambda_, psi)
        else:
            self.index = faiss.IndexBinaryFlat(vector_size)

    def _create_blocks(self):
        blocks: Dict[int, set] = defaultdict(set)
        bitarray_dict: dict = self.encoded_data.bitarray_dict
        lower_bound = self.encoded_data.bounds[0]
        bitarray_dict: Dict[int, Dict[str, bitarray]] = {k: v for k, v in bitarray_dict.items()
                                                         if lower_bound <= k}

        vector, _ = self._init_vector(bitarray_dict)


        self.distances, self.neighbors = self.index.search(vector, self.top_k) #type: ignore
        for _entity in range(0, self.neighbors.shape[0]):
            _entity_id = _entity + lower_bound
            for _neighbor_id in self.neighbors[_entity]:
                if _neighbor_id < 0:
                    break
                blocks[int(_neighbor_id)].add(_entity_id)
        return blocks

    def build_blocks(self,
                     encoded_data: BloomEncodedData,
                     attributes: List[str] = None,
                     top_k: int = 30,
                     ) -> Dict[Union[str, int], Block]:
        self.top_k = top_k

        return super().build_blocks(encoded_data, attributes)

    def _fit(self):
        enc_bitarray_dict: dict = self.encoded_data.bitarray_dict
        bitarray_dict: Dict[int, Dict[str, bitarray]] = {k: v for k, v in enc_bitarray_dict.items()
                                                         if 0 <= k < self.encoded_data.bounds[0]}
        vector, bitarray_list = self._init_vector(bitarray_dict)

        self._set_index(len(bitarray_list[0]))
        self.index.add(vector) #type: ignore



