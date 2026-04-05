"""Encoded data class file"""

import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
import itertools
import datetime
from dataclasses import dataclass

import pandas as pd
import ray
from bitarray import bitarray
import numpy as np

from privjedai.utils import  are_matching
from privjedai.evaluation import Evaluation
from privjedai.datamodel import EncodedData, PPRLFeature

@dataclass
class BloomEncodedDataMetadata:
    """Metadata for BloomEncodedData"""
    length : int
    execution_time : float
    attributes : list

class BloomEncodedData(EncodedData, PPRLFeature):
    """
    Encoded data using Bloom filters for PPRL.

    Stores entity data encoded as Bloom filters with inverted index support
    for efficient similarity computation.
    """

    _method_name = "BloomEncodedData"
    _method_info = "BloomEncodedData"
    _method_short_name = "BED"

    metadata: BloomEncodedDataMetadata
    bounds : List[int]
    inverted_index : Dict[int, Dict[str, set]]
    candidate_pairs: set
    """inverted_index :   position_of_bit : entities {D0: x,y,z,...} and {D1: x2,y2,z2,...}"""
    skip_ground_truth : bool

    def __init__(self, data: Dict[int, Dict[str, List[int]]] = None, length: int = 0):
        """Initialize BloomEncodedData with optional pre-encoded data and bit length."""
        super().__init__()
        self.inverted_index = {}
        self.skip_ground_truth = True
        if data:
            self.metadata = BloomEncodedDataMetadata(length=length,
                    execution_time=0.0, attributes=[])
            self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
            if self.encoded_dict:
                self.bitarray_dict : Dict[int, Dict[str, bitarray]] = {}
                for key, value in self.encoded_dict.items():
                    self.bitarray_dict[key] = {}
                    for attr, indices in value.items():
                        b = bitarray(length)
                        b.setall(0)
                        if indices:
                            b[indices] = True
                        self.bitarray_dict[key][attr] = b

                first_attr = next(iter(self.encoded_dict.values()), {})
                if first_attr:
                    self.metadata.attributes = list(first_attr.keys())

    def __str__(self):
        return f'{self.encoded_dict}'

    def __getstate__(self):
        """Customize what gets pickled."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Customize how to restore from pickle."""
        self.__dict__.update(state)


    def to_file(self, filename:str) :
        with open(filename, "wb+") as f:
            pickle.dump(self, f)

    def set_ground_truth(self, df: pd.DataFrame) :
        """
        Sets ground truth for evaluation
        Ground truth must be the indexes of the records


        Args:
            filename (str): Ground truth file path
        """

        self.ground_truth = df
        self.skip_ground_truth = False

    def _get_dataset(self, idx: int) -> int:
        if idx >= self.bounds[0]:
            return  1
        return 0

    def _get_key_set(self, entity_id: int,
                blooms: Dict[str, List[int]],
                adjacent_bits: int) -> list:

        chunks = defaultdict(list)
        for i, attr in enumerate(self.metadata.attributes):
            offset = i * self.metadata.length
            for bl_i in blooms[attr]:
                bit_val = bl_i + offset
                chunk_index = bit_val // adjacent_bits
                chunks[chunk_index].append(bit_val)

        block_keys = []
        for index, bits in chunks.items():
            bits.sort()
            bit_key = f'{index},' + ",".join(map(str, bits))
            block_keys.append(bit_key)

        entity_ids = [entity_id] * len(block_keys)
        return entity_ids, block_keys

    def get_entity_id_block_key_pairs(self,
            adjacent_bits: int = 1) -> Tuple[dict, dict]:
        """
        Convert to entity-bit mapping and create inverted index.
        Working only with concatenated bitarrays
        e.g if name : (0001) and email (1111)
        then will check 0001111 for as a whole bloom filter

        Args:
            attribute: Specific attribute to process, or None for all

        Returns:
            Tuple of (entity_to_bits, inverted_index)
        """

        entity_ids = []
        block_keys = []
        for idx, blooms in self.encoded_dict.items():
            ent_id, blk_keys = self._get_key_set(idx, blooms, adjacent_bits)
            entity_ids.extend(ent_id)
            block_keys.extend(blk_keys)

        return np.array(entity_ids), np.array(block_keys)


    @classmethod
    def from_file(cls, *filenames) -> "BloomEncodedData":
        """Load a new instance from file."""
        ret : BloomEncodedData = None
        dataset_num = 0
        for filename in filenames:
            with open(filename, "rb") as f:
                obj : BloomEncodedData = pickle.load(f)
                if not ret:
                    ret = obj
                    ret.bounds = [len(ret.encoded_dict)]
                    # ret.dataset_indexes[dataset_num] = (0, len(ret.encoded_dict)-1)
                else:
                    offset = len(ret.encoded_dict)
                    encoded_dict = obj.encoded_dict
                    bitarray_dict = obj.bitarray_dict
                    ret.encoded_dict = {**ret.encoded_dict,
                                **{key + offset :  value for key, value in encoded_dict.items()} }
                    ret.bitarray_dict = {**ret.bitarray_dict,
                                **{key + offset :  value for key, value in bitarray_dict.items()} }
                    ret.bounds.append(offset + len(obj.encoded_dict))

            dataset_num += 1

        ret.inverted_index = None

        return ret

    def _create_inverted_index(self) -> None:
        self.inverted_index = defaultdict(lambda : defaultdict(set))
        dataset = 0
        for i, encoded in self.encoded_dict.items():
            if  i >= self.bounds[dataset]:
                dataset += 1
            for attr_bit_positions in encoded.values():
                for bit_position in attr_bit_positions:
                    self.inverted_index[bit_position][dataset].add(i)


    def _serial_evaluate(self, entity_index: dict):
        true_positives = 0
        for _, (id1, id2) in self.ground_truth.iterrows():
            id2 = id2 + self.bounds[0]
            if id1 in entity_index and    \
            id2 in entity_index and are_matching(entity_index, id1, id2):
                true_positives += 1

        return true_positives

    def calculate_candidate_pairs(self, inverted_index: dict) -> int:
        """Returns length of candidate pairs"""
        candidate_pairs  = set()
        for block in inverted_index.values():
            candidate_pairs.update(set(
                itertools.product(block[0], block[1])))
        return len(candidate_pairs)

    def _get_set_concat_bloom_filter(self, entity_id: int) -> set:
        _set_concat_bf = set()
        for i, bloom_filter  in enumerate(
            self.encoded_dict[entity_id].values()):
            bloom_filter_concat = [(bloom_filter_index + \
                                    self.metadata.length*i)
                    for bloom_filter_index in bloom_filter]
            _set_concat_bf.update(set(bloom_filter_concat))


        return _set_concat_bf



    @ray.remote
    def _ray_get_candidate_pairs(self, index: int, adjacent_bits: int) -> list:
        adjacent_list = list(range(adjacent_bits))
        total_length = len(self.metadata.attributes) * self.metadata.length
        _entity_d1 = self._get_set_concat_bloom_filter(index)
        _entities_d2 = {key: value
                for key, value in self.encoded_dict.items()
                if key >= self.bounds[0]}

        candidate_pairs = []

        for index_d2 in _entities_d2:
            _entity_d2 = self._get_set_concat_bloom_filter(index_d2)
            for bit in range((total_length + adjacent_bits - 1) //  adjacent_bits):
                adj_list = [(i + bit * adjacent_bits) for i in adjacent_list]
                cont_a = [i for i in adj_list if i in _entity_d1]
                cont_b = [i for i in adj_list if i in _entity_d2]
                if len(cont_b) == 0 or len(cont_a) == 0:
                    continue
                if cont_a == cont_b:
                    candidate_pairs.append((index, index_d2))
                    break

        return candidate_pairs


    def calculate_average_active_bits(self) -> float:
        """Calculate average active bits"""

        total_length = len(self.metadata.attributes) * \
            self.metadata.length
        num_of_entities = self.bounds[1]

        print(f"Number_of_ entities {num_of_entities}")

        bloom_filter_matrix = np.zeros(
            (num_of_entities, total_length),
            dtype=bool)

        for entity_id, fields in self.encoded_dict.items():
            for i, bloom_filter_indices in enumerate(fields.values()):
                offset = i * self.metadata.length
                if bloom_filter_indices:
                    bloom_filter_matrix[entity_id,
                        np.array(bloom_filter_indices) + offset]  = True

        print("Bloom Filter created")

        # 4. Count Active Bits
        # Corrected: No need to loop. Numpy is optimized for this.
        active_bits = np.count_nonzero(bloom_filter_matrix)

        # 5. Return Ratio
        # This returns the density (0.0 to 1.0).
        # If you wanted average bits per entity, remove the division by total_length.
        total_possible_bits = total_length * num_of_entities

        if total_possible_bits == 0:
            return 0.0

        return int(active_bits) / int(total_possible_bits)


    def _init_bloom_filter_matrix(self, total_length: int) -> np.array:

        bloom_filter_length = self.bounds[1]
        bloom_filter_matrix = np.zeros(
            (bloom_filter_length, total_length),
            dtype=bool)


        rows = []
        cols = []
        for entity_id, fields in self.encoded_dict.items():
            for i, bloom_filter_indices in enumerate(fields.values()):
                if bloom_filter_indices:
                    offset = i * self.metadata.length
                    rows.extend([entity_id] * len(bloom_filter_indices))
                    cols.extend([idx + offset for idx in bloom_filter_indices])

        if rows:
            bloom_filter_matrix[rows, cols] = True

        return bloom_filter_matrix
    def _calculate_existing_pairs(self, entities_1: np.array,
                entities_2: np.array, adjacent_bits: int,
                total_length: int) -> np.array:

        existing_pairs = np.zeros((entities_1.shape[0], entities_2.shape[0]), dtype=bool)

        for start_bit in range(0, total_length, adjacent_bits):
            end_bit = min(start_bit + adjacent_bits, total_length)
            block_1 = entities_1[:, start_bit:end_bit]
            block_2 = entities_2[:, start_bit:end_bit]
            if not block_1.flags['C_CONTIGUOUS']:
                block_1 = np.ascontiguousarray(block_1)
            if not block_2.flags['C_CONTIGUOUS']:
                block_2 = np.ascontiguousarray(block_2)

            void_dtype = np.dtype((np.void, block_1.shape[1] * block_1.itemsize))
            view_1 = block_1.view(void_dtype).ravel()
            view_2 = block_2.view(void_dtype).ravel()
            valid_1 = block_1.any(axis=1)

            matches = (view_1[:, None] == view_2[None, :]) & valid_1[:, None]

            existing_pairs |= matches
            if existing_pairs.all():
                break
            return existing_pairs

    def calculate_distinct_candidate_pairs(self, adjacent_bits: int = 1) -> int:
        """
        Function that produces candidate pairs based on blocking of adjacent bits
        Args:
            adjacent_bits (int): Adjacent bits that are used as blocking keys

        Returns:
            list: All the candidate pairs

        """
        total_length = len(self.metadata.attributes) * \
            self.metadata.length
        bloom_filter_matrix = self._init_bloom_filter_matrix(total_length)

        entities_1 = bloom_filter_matrix[:self.bounds[0]]
        entities_2 = bloom_filter_matrix[self.bounds[0]:self.bounds[1]]


        existing_pairs = self._calculate_existing_pairs(entities_1, entities_2,
                        adjacent_bits, total_length)
        return np.count_nonzero(existing_pairs)

    def _init_bloom_filter_matrix_evaluation(self, total_length: int) -> np.array:
        bloom_filter_matrix = np.zeros(
            (len(self.encoded_dict), total_length),
            dtype=bool )

        for entity_id, fields in self.encoded_dict.items():
            for i, bloom_filter_indices in enumerate(fields.values()):
                offset = i * self.metadata.length
                if bloom_filter_indices:
                    bloom_filter_matrix[entity_id,
                        np.array(bloom_filter_indices) + offset]  = True
        return bloom_filter_matrix

    def evaluate(self,
        adjacent_bits: int = 1,
        export_to_df: bool = False,
        with_classification_report: bool = False,
        verbose: bool = True) -> any:
        """Function to evaluate encoder treating it
        as a blocking method: f1-score, recall and precision

        Args:
            export_to_df (bool) : Create evaluation dataframe
            with_classification_report (bool) : Printing Info for the blocking method
            with_stats (bool) : Printing Method's Statistics

        Returns:
            Evaluation : Evaluation Object with F1, Recall, Precision etc.
        """
        self.execution_time = 0.0
        if self.skip_ground_truth:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file."
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self)

        ground_truth_list = list(self.ground_truth.itertuples(index=False, name=None))
        total_matching_pairs = 0
        true_positives = 0
        total_length = len(self.metadata.attributes) * \
              self.metadata.length
        self.candidate_pairs = set()


        bloom_filter_matrix = self. \
            _init_bloom_filter_matrix_evaluation(total_length)


        total_matching_pairs, true_positives = self._get_matching_pairs_loop(
            bloom_filter_matrix, adjacent_bits,
            ground_truth_list)


        eval_obj.calculate_scores(true_positives=true_positives,
                                total_matching_pairs=total_matching_pairs)
        eval_result = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                with_classification_report,
                                verbose)

        return eval_result

    def _reduce_gt_array_leq_8(self,
            pairs_entities: np.array,
            ground_truth_array: np.array,
            start_1: int,
            start_2: int,) -> np.array:


        if ground_truth_array.size > 0:
            pairs_sliced = np.argwhere(pairs_entities)
            pairs_sliced = np.ascontiguousarray(pairs_sliced)
            pairs_sliced[:,0] += start_1
            pairs_sliced[:,1] += start_2 - self.bounds[0]

            mask = ~np.isin(
                ground_truth_array.view([('', ground_truth_array.dtype)] * 2),
                pairs_sliced.view([('', pairs_sliced.dtype)] * 2)
            ).ravel()

            ground_truth_array = ground_truth_array[mask]

        return ground_truth_array


    def _increase_total_matching_pairs_leq_8(self,
            total_matching_pairs: int,
            entities_1_int: np.array,
            entities_2_int: np.array) -> int:

        equal_entities = (entities_1_int != 0) & (entities_1_int == entities_2_int)
        pairs_entities = np.count_nonzero(equal_entities, axis=-1)
        total_matching_pairs += np.count_nonzero(equal_entities)


        return total_matching_pairs, pairs_entities

    @staticmethod
    def _get_entities_int_leq_8(bloom_filter_matrix: np.array,
                batch_info: tuple,
                adjacent_bits: int, n_slices: int,
                bound: int) -> np.array:
        batch_start, batch_size = batch_info
        batch_end = min(batch_start +  batch_size, bound)
        entities = bloom_filter_matrix[batch_start:batch_end]. \
            reshape(batch_end - batch_start,
                     n_slices, adjacent_bits)

        entities_int = np.packbits(entities, axis=-1,
                                    bitorder='little').squeeze(-1)
        return entities_int

    def _get_matching_pairs_loop_leq_8(self,
                bloom_filter_matrix: np.array,
                ground_truth_array: np.array,
                adjacent_bits: int,
                true_positives: int,
                ) -> Tuple[int, int]:

        batch_size = 2_000
        entity_1 : np.array = bloom_filter_matrix[0]
        len_entity_1 = len(entity_1)
        total_matching_pairs = 0
        for start_1 in range(0, self.bounds[0], batch_size):
            print(f'{start_1}/{self.bounds[0]} {datetime.datetime.now()}')
            n_slices = len_entity_1 // adjacent_bits
            entities_1_int = self._get_entities_int_leq_8(
                bloom_filter_matrix, (start_1, batch_size),
                adjacent_bits, n_slices, self.bounds[0])
            entities_1_int = entities_1_int[:, None, :]
            for start_2 in range(self.bounds[0], self.bounds[1], batch_size):
                entities_2_int = self._get_entities_int_leq_8(
                    bloom_filter_matrix, (start_2, batch_size),
                    adjacent_bits, n_slices, self.bounds[1])

                total_matching_pairs, pairs_entities = self._increase_total_matching_pairs_leq_8(
                    total_matching_pairs, entities_1_int, entities_2_int)
                ground_truth_array = self._reduce_gt_array_leq_8(pairs_entities, ground_truth_array,
                        start_1, start_2)

        true_positives -= len(ground_truth_array)
        return int(total_matching_pairs), int(true_positives)

    def _get_matching_pairs_loop_gt_8(self,
            bloom_filter_matrix: np.array,
            adjacent_bits: int,
            ground_truth_array: np.array,
            true_positives: int,) -> Tuple[int, int]:

        total_matching_pairs = 0
        len_entity_1 = len(bloom_filter_matrix[0])

        for start_bit in range(0, len_entity_1, adjacent_bits):
            end_bit = min(start_bit + adjacent_bits, len_entity_1)
            enitities_1 = bloom_filter_matrix[0:self.bounds[0], None, start_bit:end_bit]
            masked_1 = enitities_1.any(axis=-1)
            enitities_2 = bloom_filter_matrix[None, self.bounds[0]:self.bounds[1],
                            start_bit:end_bit]
            equal_entities = ~np.any(enitities_1 != enitities_2, axis = -1) & masked_1
            total_matching_pairs += np.count_nonzero(equal_entities)
            if ground_truth_array.size > 0:
                pairs_sliced = np.argwhere(equal_entities)
                # Create mask of ground_truth pairs that are NOT matched
                pairs_sliced = np.ascontiguousarray(pairs_sliced)

                mask = ~np.isin(
                    ground_truth_array.view([('', ground_truth_array.dtype)] * 2),
                    pairs_sliced.view([('', pairs_sliced.dtype)] * 2)
                ).ravel()

                ground_truth_array = ground_truth_array[mask]


        true_positives -= len(ground_truth_array)


        return int(total_matching_pairs), int(true_positives)


    def _get_matching_pairs_loop(self, bloom_filter_matrix: np.array,
                            adjacent_bits: int, ground_truth_list):


        true_positives = len(ground_truth_list)
        ground_truth_array = np.array(ground_truth_list)
        ground_truth_array = np.ascontiguousarray(ground_truth_array)

        if adjacent_bits <= 8:
            return self._get_matching_pairs_loop_leq_8(
                bloom_filter_matrix, ground_truth_array,
                adjacent_bits,
                true_positives)

        return self._get_matching_pairs_loop_gt_8(
            bloom_filter_matrix, adjacent_bits,
            ground_truth_array, true_positives)



    # # NOTE: Delete in the near future
    # def evaluate(self,
    #         adjacent_bits: int = 1,
    #         batch_size: int = 100,
    #         export_to_df: bool = False,
    #         with_classification_report: bool = False,
    #         verbose: bool = True) -> any:
    #     """Function to evaluate encoder treating it
    #     as a blocking method: f1-score, recall and precision

    #     Args:
    #         export_to_df (bool) : Create evaluation dataframe
    #         with_classification_report (bool) : Printing Info for the blocking method
    #         with_stats (bool) : Printing Method's Statistics

    #     Returns:
    #         Evaluation : Evaluation Object with F1, Recall, Precision etc.
    #     """
    #     if self.skip_ground_truth:
    #         raise AttributeError("Can not proceed to evaluation without a ground-truth file."
    #                 "Data object has not been initialized with the ground-truth file")

    #     eval_obj = Evaluation(self)

    #     ground_truth_list = list(self.ground_truth.itertuples(index=False, name=None))
    #     total_matching_pairs = 0
    #     true_positives = 0


    #     if not ray.is_initialized():
    #         logging.getLogger("ray").setLevel(logging.ERROR)
    #         ray.init(log_to_driver=False, logging_level=logging.ERROR)

    #     encoded_dict_ref = ray.put(self.encoded_dict)

    #     total_length = len(self.attributes) * self.length

    #     num_actors = 15

    #     actors = []
    #     set_encoded_dict = []

    #     for _ in range(num_actors):
    #         actor = ConfusionMatrixRay.remote(self.length,
    #                             ground_truth_list,
    #                             adjacent_bits, total_length)
    #         future = actor.set_encoded_dict.remote(encoded_dict_ref, self.bounds[0])
    #         actors.append(actor)
    #         set_encoded_dict.append(future)

    #     ray.get(set_encoded_dict)


    #     actor_index = 0


    #     futures = []
    #     for i in range(0, self.bounds[0], batch_size):
    #         actor = actors[actor_index % num_actors]
    #         future = actor.calculate.remote(i, batch_size)
    #         futures.append(future)
    #         actor_index += 1

    #     results = ray.get(futures)
    #     for total, posititves in results:
    #         total_matching_pairs += total
    #         true_positives += posititves

    #     eval_obj.calculate_scores(true_positives=true_positives,
    #                             total_matching_pairs=total_matching_pairs)
    #     eval_result = eval_obj.report(self.method_configuration(),
    #                             export_to_df,
    #                             with_classification_report,
    #                             verbose)

    #     return eval_result

    def _configuration(self):
        return {}


class HomomorphicEcnodedData(EncodedData):
    """
    Encoded data using homomorphic encryption for PPRL.

    Stores entity data encoded with homomorphic encryption schemes,
    allowing computation on encrypted data without decryption.

    NOT YET IMPLEMENTED
    """
    def __init__(self, data: Dict[int, Dict[str, List[int]]]):
        self.encoded_dict : Dict[int, Dict[str, List[int]]] = data
        if self.encoded_dict:
            first_attr = next(iter(self.encoded_dict.values()), {})
            if first_attr:
                self.attributes = list(first_attr.keys())

    def __str__(self):
        return f'{self.encoded_dict}'
