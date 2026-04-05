"""PPRL Blocking Methods"""
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



    def _configuration(self) -> dict:  #pragma: no cover
        return {}

    @staticmethod
    def _clean_blocks(blocks: dict):
        new_blocks = {}
        for key, block in blocks.items():
            if len(block) != 0:
                new_blocks[key] = block
        return new_blocks


class AbstractBlockBuilding(AbstractBlockProcessing):
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str
    _method_short_name: str

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

            if idx < self.encoded_data.bounds[0]:
                for key in record_keys:
                    blocks_d0[key].append(idx)
            else:
                for key in record_keys:
                    blocks_d1[key].append(idx)

        candidate_pairs = defaultdict(set)

        common_keys = blocks_d0.keys() & blocks_d1.keys()

        for key in common_keys:
            d1_candidates = blocks_d1[key]
            for d0_id in blocks_d0[key]:
                candidate_pairs[d0_id].update(d1_candidates)

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

        blocks: Dict[int, set] = self._create_blocks()

        self.original_num_of_blocks = len(blocks)
        self.blocks = self._clean_blocks(blocks)
        self.execution_time = time.time() - _start_time

        return self.blocks

    # def _clean_blocks(self, blocks: dict):
    #     cleaned_blocks = {}
    #     for key, block in blocks.items():
    #         if 0 not in block or 1 not in block:
    #             continue
    #         cleaned_blocks[key] = block
    #
    #     return cleaned_blocks

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
    ):  # pylint: disable=too-many-positional-arguments disable=too-many-arguments
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

    def _block_record(self, bf: bitarray) -> List[int]:
        block_keys = []
        for i, table_indices in enumerate(self.hash_indices):
            vals = (bf[idx] for idx in table_indices)
            table_block = sum(b << j for j, b in enumerate(vals))
            block_keys.append(table_block * len(self.hash_indices) + i)

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

        init_vector = np.array([np.frombuffer(b.tobytes(), dtype=np.uint8)
                           for b in bitarray_list], dtype=np.uint8)

        return init_vector, bitarray_list


    def configure_hsnw(self, hnsw_m : int = 32):
        self.configuration['hnsw_m'] = hnsw_m

    def _set_index(self, vector_size: int):
        if 'hnsw' == self.configuration['index_type']:
            self.index = faiss.IndexBinaryHNSW(vector_size,
                                   self.configuration.get('hnsw_m', 32))
            self.index.metric_type = faiss.METRIC_Jaccard
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



