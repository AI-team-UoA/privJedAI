"""Block Cleaning Module
Contains:
 - BlockFiltering
 - BlockPurging
"""
from time import time
from typing import Tuple

import numpy as np

from privjedai.blocking import AbstractBlockProcessing
from privjedai.datamodel import EncodedData
from privjedai.encoded_data import BloomEncodedData

class AbstractBlockCleaning(AbstractBlockProcessing):

    def __init__(self) -> None:
        super().__init__()

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nParameters: \n" + ''.join(['\t{0}: {1}\n'.format(k, v) for k, v in self._configuration().items()]) +
            "Runtime: {:2.4f} seconds".format(self.execution_time)
        )

    @staticmethod
    def _sorted_cardinalities(blocks_with_keys: np.ndarray, limit_: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_keys, inverse_indices = np.unique(blocks_with_keys[:,0], return_inverse=True)

        is_e1 = blocks_with_keys[:,1] < limit_
        is_e2 = ~is_e1

        count_e1 = np.bincount(inverse_indices, weights=is_e1).astype(int)
        count_e2 = np.bincount(inverse_indices, weights=is_e2).astype(int)
        cardinalities = count_e1 * count_e2
        sizes = count_e1 + count_e2


        pair_cards = cardinalities[inverse_indices]
        pair_sizes = sizes[inverse_indices]
        sort_idx = np.lexsort((pair_cards, blocks_with_keys[:,1]))

        return sort_idx, pair_cards, pair_sizes


class BlockFiltering(AbstractBlockCleaning):
    """Retains every entity in a subset of its smallest blocks.

        Filtering consists of 3 steps:
        - Blocks sort in ascending cardinality
        - Creation of Entity Index: inversed block dictionary
        - Retain every entity in ratio % of its smallest blocks
        - Blocks reconstruction
    """

    _method_name = "Block Filtering"
    _method_short_name: str = "BF"
    _method_info = "Retains every entity in a subset of its smallest blocks."

    num_of_blocks_dropped: int

    def __init__(self, ratio: float = 0.8) -> None:
        super().__init__()
        if ratio > 1.0 or ratio < 0.0:
            raise AttributeError("Ratio is a number between 0.0 and 1.0")
        else:
            self.ratio = ratio
        # self.entity_index: dict

    def __str__(self) -> str:
        print(self._method_name + self._method_info)
        print("Ratio: ", self.ratio)
        return super().__str__()

    def process(
            self,
            encoded_data: BloomEncodedData
    ) -> np.ndarray | None:
        """Main method of Block Filtering.

        Args:
            encoded_data (EncodedData): input dataset.

        Returns:
            Tuple[dict, dict]: dict of keys to Blocks, entity index (reversed blocks)
        """

        if encoded_data.blocks_with_keys is None:
            raise AttributeError("Must firstly build blocks!")


        start_time = time()
        self.encoded_data = encoded_data
        blocks_with_keys = encoded_data.blocks_with_keys

        sort_idx, _, _= self._sorted_cardinalities(blocks_with_keys, encoded_data.bounds[0])

        sorted_ids = blocks_with_keys[sort_idx, 1]
        sorted_keys = blocks_with_keys[sort_idx, 0]

        unique_sorted_ids, counts = np.unique(sorted_ids, return_counts=True)
        start_indices = np.insert(np.cumsum(counts)[:-1], 0, 0)

        ranks = np.arange(sorted_ids.shape[0]) - np.repeat(start_indices, counts)
        group_counts = np.repeat(counts, counts)

        thresholds = np.ceil(group_counts * self.ratio)

        mask = ranks < thresholds

        final_ids = sorted_ids[mask]
        final_sorted_keys = sorted_keys[mask]



        blocks_with_keys = np.column_stack([final_sorted_keys, final_ids])
        blocks_with_keys = self._clean_blocks_with_keys(blocks_with_keys, encoded_data.bounds[0])

        self.num_of_blocks_dropped = encoded_data.blocks_with_keys.shape[0] - blocks_with_keys.shape[0]
        self.execution_time = time() - start_time

        self.blocks_with_keys = blocks_with_keys
        encoded_data.set_blocks_with_keys(blocks_with_keys)

        return self.blocks_with_keys

    def _configuration(self) -> dict:
        return {
            "Ratio" : self.ratio
        }

class BlockPurging(AbstractBlockCleaning):
    """Discards the blocks exceeding a certain number of comparisons.
    """

    _method_name = "Block Purging"
    _method_short_name: str = "BP"
    _method_info = "Discards the blocks exceeding a certain number of comparisons."
    blocks_with_keys: np.ndarray
    encoded_data: BloomEncodedData
    num_of_blocks_dropped: int

    def __init__(self, smoothing_factor: float = 1.025):
        super().__init__()
        self.smoothing_factor: float = smoothing_factor
        self.max_comparisons_per_block: float

    def process(
            self,
            encoded_data: BloomEncodedData,
    ) -> np.ndarray:
        """Main method of Block Purging.

        Args:
            encoded_data (BloomEncodedData): Data module. Contains all the information about the dataset.

        Returns:
            dict: Purged blocks.
        """
        start_time = time()


        if encoded_data.blocks_with_keys is None:
            raise AttributeError("Must firstly build blocks!")


        self.blocks_with_keys = encoded_data.blocks_with_keys
        self.encoded_data = encoded_data

        new_blocks_with_keys = self._set_threshold_and_prune()
        self.num_of_blocks_dropped = self.blocks_with_keys.shape[0] - new_blocks_with_keys.shape[0]
        self.execution_time = time() - start_time
        self.blocks_with_keys = new_blocks_with_keys

        encoded_data.set_blocks_with_keys(new_blocks_with_keys)

        return new_blocks_with_keys


    def _set_threshold_and_prune(self) -> np.ndarray:
        """Calculates the maximum number of comparisons per block, so in the next step to be purged.
        """

        sort_idx, pair_card, pair_sizes = self._sorted_cardinalities(self.blocks_with_keys, self.encoded_data.bounds[0])
        # sort_idx = np.lexsort([pair_card, self.blocks_with_keys[:, 0]])


        pair_card = pair_card[sort_idx]
        pair_sizes = pair_sizes[sort_idx]

        sorted_blocks = self.blocks_with_keys[sort_idx]
        _, first_block_index = np.unique(sorted_blocks[:,0], return_index=True)


        block_cardinalities = pair_card[first_block_index]
        block_sizes = pair_sizes[first_block_index]

        cardinalities_idx = np.argsort(block_cardinalities)

        block_cardinalities = block_cardinalities[cardinalities_idx]
        block_sizes = block_sizes[cardinalities_idx]
        comparisons_level, indices = np.unique(block_cardinalities, return_inverse=True)



        block_assignments_per_comparison = np.bincount(indices, weights=block_sizes.astype(np.float64))
        total_comparisons_per_level_per_comparison = np.bincount(indices,
                                                                 weights=block_cardinalities.astype(np.float64))


        block_assignments = np.cumsum(block_assignments_per_comparison)
        total_comparisons_per_level = np.cumsum(total_comparisons_per_level_per_comparison)


        current_bc = current_cc = current_size = previous_size = 0
        for i in range(len(block_assignments)-1, 0, -1):
            previous_size = current_size
            previous_bc = current_bc
            previous_cc = current_cc
            current_size = comparisons_level[i]
            current_bc = block_assignments[i]
            current_cc = total_comparisons_per_level[i]
            if current_bc * previous_cc < self.smoothing_factor * current_cc * previous_bc:
                break

        self.max_comparisons_per_block = previous_size

        # print(f"Max comparisons per block: {self.max_comparisons_per_block} and smoothing factor: {self.smoothing_factor}")

        mask = pair_card <= self.max_comparisons_per_block

        return sorted_blocks[mask]


    def _configuration(self) -> dict:
        return {
            "Smoothing factor" : self.smoothing_factor,
            "Max Comparisons per Block" : self.max_comparisons_per_block
        }

