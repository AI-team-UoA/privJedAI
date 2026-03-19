"""PPRL Blocking Methods"""
import random
from collections import defaultdict
from typing import List, Literal, Dict, Union
import time
from abc import abstractmethod

import faiss
from bitarray import bitarray
import numpy as np
from privjedai.datamodel import  Block, PPRLFeature, EncodedData
from privjedai.encoded_data import BloomEncodedData
from privjedai.evaluation import Evaluation


class AbstractBlockProcessing(PPRLFeature):
    """
    Abstract Class for blocking
    """
    def __init__(self):
        super().__init__()
        self.blocks : dict
        self.attributes : list
        self.encoded_data : EncodedData = None

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        if not self.encoded_data:
            raise AttributeError("Encoded data must be instantiated")
        _configuration = self._configuration()
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            ("\nParameters: \n" + ''.join([f'\t{k}: {v}\n' for k, v in _configuration.items()])
            if self._configuration().items() else "\nParameters: Parameter-Free method\n") +
            "Attributes:\n\t" + ', '.join(c for c in self.attributes) +
            f"\nRuntime: {self.execution_time:2.4f} seconds"
        )



    def evaluate(self,
            prediction,
            export_to_df: bool = False,
            with_classification_report: bool = False,
            verbose: bool = True) -> any:
        """Function to evaluate blocking methods f1-score, recall and precision

        Args:
            prediction (dict) : Blocks predicted from the blocking method
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """
        if prediction is None:
            if self.blocks is None:
                raise AttributeError("Can not proceed to evaluation without build_blocks.")
            eval_blocks = self.blocks
        else:
            eval_blocks = prediction

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.encoded_data)

        ground_truth_list = list(self.encoded_data.
                        ground_truth.itertuples(index=False, name=None))


        candidate_pairs = {
            (id1, id2)
            for block in eval_blocks.values()
            for id1 in block[0]
            for id2 in block[1]
        }


        total_matching_pairs = len(candidate_pairs)
        offset = self.encoded_data.bounds[0]
        true_positives = sum(
            1 for id1, id2 in candidate_pairs
            if (id1, id2-offset) in ground_truth_list
        )

        eval_obj.calculate_scores(true_positives=true_positives,
                                total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

        return eval_result

    def _configuration(self)-> dict:            #pragma: no cover
        return {}

class AbstractBlockBuilding(AbstractBlockProcessing):
    """Abstract class for the block building method
    """

    _method_name: str
    _method_info: str
    _method_short_name: str


    def __init__(self, seed : int = 42):
        super().__init__()
        self.seed : int = seed
        self.original_num_of_blocks : int
        self.blocks : dict = None

    def _create_blocks(self) -> Dict[str, Dict[int, set]]:
        blocks : Dict[str, Dict[int, set]] = {}
        dataset = 0
        for idx, bloom_filters in self.encoded_data.bitarray_dict.items():
            if idx >= self.encoded_data.bounds[dataset]:
                dataset += 1
            concatenated_bitarray = sum(
                (array for attr, array in bloom_filters.items()
                if attr in self.attributes), bitarray())

            for key in self._block_record(concatenated_bitarray):
                blocks.setdefault(str(key), {})
                blocks[str(key)].setdefault(dataset, set())
                blocks[str(key)][dataset].add(idx)
        return blocks



    def build_blocks(
            self,
            encoded_data: BloomEncodedData,
            attributes : List[str] = None,
    ) -> Dict[Union[str, int], Block]:
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
        self.encoded_data : BloomEncodedData = encoded_data
        bloom_attributes = encoded_data.metadata.attributes

        if attributes:
            if set(attributes) > set(bloom_attributes):
                raise ValueError(f"Attributes must be a subset "
                            f"of the attributes in Bloom Filters : {bloom_attributes}")
            self.attributes = attributes
        else:
            self.attributes = bloom_attributes

        self._fit()

        blocks : Dict[str, Dict[int, set]] = self._create_blocks()
        self.original_num_of_blocks = len(blocks)
        self.blocks = blocks
        self.blocks = self._clean_blocks(blocks)
        self.execution_time = time.time() - _start_time

        return self.blocks

    def _clean_blocks(self, blocks: dict):
        cleaned_blocks = {}
        for key, block in blocks.items():
            if 0 not in block or 1 not in block:
                continue
            cleaned_blocks[key] = block

        return cleaned_blocks


    @abstractmethod
    def _fit(self) -> None:
        pass  # pragma: no cover

    def _block_record(self, bf: bitarray) -> List[str]:
        pass # pragma: no cover

    def _configuration(self) -> dict:
        return {}   # pragma: no cover



class LSHBlocker(AbstractBlockBuilding):
    """
    Implements Locality-Sensitive Hashing (LSH)-based blocking for
    Bloom-filter–encoded records.

    Each encoded record produces Λ (lambda) blocking keys, each of
    length Ψ (psi) bits. These keys are used to group similar records
    together before performing expensive pairwise comparisons.

    Attributes:
        psi (int): Number of bit positions per key (Ψ).
        lmbda (int): Number of blocking keys per record (Λ).
        prune_ratio (float): Ratio defining how frequently occurring
            or rare bit positions are pruned from the candidate pool.
        prune_sample (int): Number of records sampled to estimate bit
            frequency distribution.
        seed (int): Random seed for reproducibility.
    """
    _method_name = "LSHBlocker"
    _method_info = "LSHBlocker"
    _method_short_name = "LSHBlocker"

    def __init__(
        self,
        psi: int = 36,
        lmbda: int = 3,
        prune_ratio: float = 0.6,
        prune_sample: int = 1000,
        seed: int = 42
    ):                                          # pylint: disable=too-many-positional-arguments disable=too-many-arguments
        """
        Initialize LSHBlocker

        Args:
            psi (int): number of bit positions per key (Ψ)
            lmbda (int): number of keys per record (Λ)
            prune_ratio (float): ratio of most frequent/uncommon bit positions to prune
            prune_sample (int): number of records to sample for frequency estimation
        """
        super().__init__(seed=seed)
        if psi < 1 or lmbda < 1 :
            raise ValueError(f"Both Values psi and lmbda must be positive numbers : {psi}, {lmbda}")

        self._rng : random.Random
        self.psi = psi
        self.lmbda = lmbda
        self.prune_ratio = prune_ratio
        self.prune_sample = prune_sample
        self._bit_positions = None  # will hold usable bit indices
        self._encoded_data : BloomEncodedData

    def _configuration(self):
        return {"psi" : self.psi,
            "lmbda": self.lmbda,
            "prune_ratio": self.prune_ratio,
            "prune_sample": self.prune_sample,
            "seed": self.seed}


    def _select_bit_positions(self, bflist: List[bitarray]) -> List[int]:
        """
        Determine a candidate pool of bit positions excluding too frequent or rare bits.
        Frequency estimated from initial sample.
        """
        m = len(bflist[0])
        freq = np.zeros(m, dtype=int)
        sample = bflist if len(bflist) <= self.prune_sample else bflist[: self.prune_sample]
        for bf in sample:
            freq += np.array(bf.tolist(), dtype=int)

        # normalize to frequency ratio
        freq_ratio = freq / len(sample)

        mask = (freq_ratio > self.prune_ratio) | (freq_ratio < (1 - self.prune_ratio))
        # candidate positions are those not masked
        # print(mask)
        candidates = [i for i, bad in enumerate(mask) if not bad]

        if len(candidates) < self.psi:
            raise ValueError("Not enough bit positions after pruning")
        return candidates


    # Change fit to work with concatenated bloom_filters
    def _fit(self):
        """Initialize blocker by sampling to prune and selecting usable bit positions."""
        bitarray_dict : Dict[int, Dict[str, bitarray]] = self.encoded_data.bitarray_dict

        bitarray_list: List[bitarray] = [
            sum((array for attr, array in attr_bittarays.items()
                if attr in self.attributes), bitarray())
                for attr_bittarays in bitarray_dict.values()
            ]

        self._rng = random.Random(self.seed)
        self._bit_positions = self._select_bit_positions(bitarray_list)

    def _block_record(self, bf: bitarray) -> List[str]:
        """
        Generate Λ blocking keys for a single Bloom filter record.
        Each key is Ψ bits sampled from bf at selected positions.
        """
        keys = []
        for _ in range(self.lmbda):
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
    lmbda : int, default=3
        Number of blocking keys per record.
    seed : int, default=42
        Random seed for reproducibility.
    """

    _method_name = "BitBlocker"
    _method_info = "BitBlocker"
    _method_short_name = "BitBlocker"



    def __init__(self,
            psi: int = 36,
            lmbda: int = 3,
            seed : int = 42):
        """
        Initializer of BitBlocker

        Args:
            psi (int): Number of bits per blocking key
            lmbda (int): Number of blocking keys per record
            seed (int): Random seed for reproducibility

        """
        super().__init__(seed=seed)
        if psi < 1 or lmbda < 1 :
            raise ValueError(f"Both Values psi and lmbda must be positive numbers : {psi}, {lmbda}")

        self.psi = psi
        self.lmbda = lmbda
        self.hash_len : int
        self.encoded_data : BloomEncodedData
        self.rng : random.Random
        self.hash_indices : tuple

    def _fit(self) -> None:
        if self.encoded_data.metadata.length % 4 != 0 :
            raise ValueError("Bloom Filters' length must multiple of 4.")

        self.hash_len = self.encoded_data.metadata.length * len(self.attributes)
        self.rng = random.Random(self.seed)
        self.hash_indices = tuple(self.rng.sample(range(self.hash_len), self.psi)
                                  for _ in range(self.lmbda))


    def _block_record(self, bf: bitarray) -> List[int]:
        block_keys = []
        for i, table_indices in enumerate(self.hash_indices):
            vals = map(bf.__getitem__, table_indices)
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
        vector_size (int): Dimensionality of the input vectors.
        top_k (int): Number of nearest neighbors to retrieve for each entity.
        index (faiss.IndexBinaryFlat): FAISS binary index for efficient similarity search.
        neighbors (np.array): Array containing neighbor indices from FAISS search.
        distances (np.array): Array containing distances to neighbors from FAISS search.
    """

    _method_name = "FAISS Blocking"
    _method_short_name: str = "FAISS"
    _method_info = "Faiss blocking."


    def __init__(self, index_type: Literal['flat','hnsw','lsh'] = 'flat'):
        super().__init__()
        self.top_k: int
        self.encoded_data : BloomEncodedData
        self.neighbors : np.array
        self.distances : np.array
        self.index : faiss.Index
        self.index_type = index_type



    def _set_index(self, vector_size: int):
        if 'hnsw' == self.index_type:
            self.index = faiss.IndexBinaryHNSW(vector_size, 32)
            self.index.metric_type = faiss.METRIC_Jaccard
        elif 'lsh' == self.index_type:
            lmbda : int = 8
            psi = vector_size // lmbda
            self.index = faiss.IndexBinaryMultiHash(vector_size, lmbda, psi)
        else:
            self.index = faiss.IndexBinaryFlat(vector_size)



    def _create_blocks(self):
        blocks : Dict[int, set] = defaultdict(set)
        bitarray_dict : dict = self.encoded_data.bitarray_dict
        lower_bound = self.encoded_data.bounds[0]
        bitarray_dict : Dict[int, Dict[str, bitarray]] = {k : v for k,v in bitarray_dict.items()
                    if lower_bound <= k }


        bitarray_list: List[bitarray] = [
            sum((array for attr, array in attr_bittarays.items()
                if attr in self.attributes), bitarray())
                for attr_bittarays in bitarray_dict.values()
        ]

        vector = np.array([np.frombuffer(b.tobytes(), dtype=np.uint8)
            for b in bitarray_list], dtype=np.uint8)


        self.distances, self.neighbors = self.index.search(vector, self.top_k)
        for _entity in range(0, self.neighbors.shape[0]):
            _entity_id = _entity + lower_bound
            for _neighbor_id in self.neighbors[_entity]:
                if _neighbor_id < 0:
                    break
                blocks[int(_neighbor_id)].add(_entity_id)
        return blocks

    def build_blocks(self,
                    encoded_data: BloomEncodedData,
                    attributes : List[str] = None,
                    top_k: int = 30,
    ) -> Dict[Union[str, int], Block]:
        self.top_k = top_k

        return super().build_blocks(encoded_data, attributes)



    def _fit(self):
        enc_bitarray_dict : dict = self.encoded_data.bitarray_dict
        bitarray_dict : Dict[int, Dict[str, bitarray]] = {k : v for k,v in enc_bitarray_dict.items()
                        if 0 <= k < self.encoded_data.bounds[0]}
        bitarray_list: List[bitarray] = [
            sum((array for attr, array in attr_bittarays.items()
                if attr in self.attributes), bitarray())
                for attr_bittarays in bitarray_dict.values()
        ]

        vector = np.array([np.frombuffer(b.tobytes(), dtype=np.uint8)
            for b in bitarray_list], dtype=np.uint8)

        self._set_index(len(bitarray_list[0]))
        self.index.add(vector)

    def _clean_blocks(self, blocks):
        new_blocks = {}
        for key, block in blocks.items():
            if len(block) != 0:
                new_blocks[key] = block
        return blocks

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

        if self.encoded_data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.encoded_data.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                        "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.encoded_data)
        true_positives = 0
        total_matching_pairs = 0
        for block in prediction.values():
            total_matching_pairs += len(block)

        for _, (id1, id2) in self.encoded_data.ground_truth.iterrows():
            id2 = self.encoded_data.bounds[0] + id2
            if id1 in prediction and id2 in prediction[id1]:
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives,
                                  total_matching_pairs=total_matching_pairs)
        return eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)
