"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
from collections import defaultdict
import numpy as np

from privjedai.base.evaluation import BaseEvaluation


class Evaluation(BaseEvaluation):
    """Evaluation class. Contains multiple methods for all the fitted & predicted data.
    """

    def evaluate_candidate_pairs(self, prediction: dict) -> None:
        """
        Evaluates the candidate pairs based on the predicted matches.
        Args:
            prediction:  Dict   A dictionary where keys are candidate pair IDs and values are lists of predicted matches. Returns:
        """
        total_matching_pairs = 0
        bounds_offset = self.encoded_data.bounds[0]

        for block in prediction.values():
            total_matching_pairs += len(block)

        true_positives = sum(
            1 for id1, id2 in self.encoded_data.ground_truth.values
            if id1 in prediction and (id2 + bounds_offset) in prediction[id1]
        )
        self.calculate_scores(true_positives, total_matching_pairs)

    def evaluate_unique_blocks(self, blocks_with_keys: np.ndarray, limit_: int):
        if blocks_with_keys.shape[0] == 0:
            self.calculate_scores(0, 0)
            return

        mask = blocks_with_keys[:,1] < limit_
        entity_1_keys = np.unique(blocks_with_keys[mask], axis=0)
        entity_2_keys = np.unique(blocks_with_keys[~mask], axis=0)


        if entity_1_keys.size == 0 or entity_2_keys.size == 0:
            self.calculate_scores(0, 0)
            return

        b1, c1 = np.unique(entity_1_keys[:, 0], return_counts=True)
        b2, c2 = np.unique(entity_2_keys[:, 0], return_counts=True)
        common_blocks, ind1, ind2 = np.intersect1d(b1, b2, return_indices=True)

        if common_blocks.size == 0:
            return self.calculate_scores(0, 0)


        c1, c2 = c1[ind1], c2[ind2]
        entity_1_keys = entity_1_keys[np.isin(entity_1_keys[:, 0], common_blocks)]
        entity_2_keys = entity_2_keys[np.isin(entity_2_keys[:, 0], common_blocks)]
        d1_groups = np.split(entity_1_keys[:, 1], np.cumsum(c1)[:-1])
        d2_groups = np.split(entity_2_keys[:, 1], np.cumsum(c2)[:-1])

        total_d1_entities = np.unique(entity_1_keys[:,1]).size
        total_d2_entities = np.unique(entity_2_keys[:,1]).size

        max_possible_pairs = total_d1_entities * total_d2_entities

        # 6. --- CHUNKED CARTESIAN PRODUCT & AGGREGATION ---
        unique_candidate_pairs = np.array([], dtype=np.uint64)
        pair_buffer = []
        buffer_size = 0
        FLUSH_LIMIT = 10_000_000  # Flush memory every 10M pairs

        for g1, g2 in zip(d1_groups, d2_groups):
            # Create a grid for the current block
            e1_grid, e2_grid = np.meshgrid(g1, g2, indexing='ij')

            # Pack (e1, e2) into a single uint64 integer: (e1 << 32) | e2
            packed_pairs = (e1_grid.ravel().astype(np.uint64) << 32) | e2_grid.ravel().astype(np.uint64)

            pair_buffer.append(packed_pairs)
            buffer_size += packed_pairs.size

            # Periodically deduplicate to prevent memory explosions
            if buffer_size >= FLUSH_LIMIT:
                batch = np.concatenate(pair_buffer)
                unique_candidate_pairs = np.unique(np.concatenate([unique_candidate_pairs, batch]))
                pair_buffer = []
                buffer_size = 0

                if unique_candidate_pairs.size == max_possible_pairs:
                    return self.calculate_scores(self.encoded_data.ground_truth.values.shape[0],
                                                 max_possible_pairs)

        # Flush remaining items in the buffer
        if pair_buffer:
            batch = np.concatenate(pair_buffer)
            unique_candidate_pairs = np.unique(np.concatenate([unique_candidate_pairs, batch]))

        total_matching_blocks = unique_candidate_pairs.size

        if total_matching_blocks == max_possible_pairs:
            return self.calculate_scores(self.encoded_data.ground_truth.values.shape[0],
                                         total_matching_blocks)


        # 7. --- TRUE POSITIVES CALCULATION ---
        # Pack the ground truth values using the same bitshift logic
        gt_array = np.ascontiguousarray(self.encoded_data.ground_truth.values)
        gt_array[:, 1] += self.encoded_data.bounds[0]
        gt_packed = (gt_array[:, 0].astype(np.uint64) << 32) | gt_array[:, 1].astype(np.uint64)

        # Intersect the two 1D uint64 arrays
        true_positives = np.intersect1d(unique_candidate_pairs, gt_packed, assume_unique=True).size

        return self.calculate_scores(true_positives, total_matching_blocks)


    def evaluate_blocks(self, blocks_with_keys: np.ndarray, limit_: int):

        unique_keys, inverse_keys = np.unique(blocks_with_keys[:, 0],  return_inverse=True)
        minlength = unique_keys.shape[0]


        entity_1_keys = inverse_keys[blocks_with_keys[:, 1] < limit_]
        entity_2_keys = inverse_keys[blocks_with_keys[:, 1] >= limit_]


        cnt_entity_1_blocks = np.bincount(entity_1_keys, minlength=minlength)
        cnt_entity_2_blocks = np.bincount(entity_2_keys, minlength=minlength)

        blocks_cardinalities = cnt_entity_1_blocks * cnt_entity_2_blocks
        total_matching_blocks = np.sum(blocks_cardinalities)

        # total_matching_blocks = blocks_with_keys.shape[0]
        id_to_keys = defaultdict(set)
        for key, id_ in blocks_with_keys:
            id_to_keys[id_].add(key)

        true_positives = 0
        bounds_offset = self.encoded_data.bounds[0]

        for id1, id2 in self.encoded_data.ground_truth.values:
            if id1 in id_to_keys and (id2 + bounds_offset) in id_to_keys:
                if not id_to_keys[id1].isdisjoint(id_to_keys[id2 + bounds_offset]):
                    true_positives += 1

        self.calculate_scores(true_positives, total_matching_blocks)
