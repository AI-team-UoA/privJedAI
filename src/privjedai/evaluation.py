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
            self.calculate_scores(0, 0)
            return

        c1, c2 = c1[ind1], c2[ind2]
        entity_1_keys = entity_1_keys[np.isin(entity_1_keys[:, 0], common_blocks)]
        entity_2_keys = entity_2_keys[np.isin(entity_2_keys[:, 0], common_blocks)]

        repeats_for_d1 = np.repeat(c2, c1)
        e1_pairs = np.repeat(entity_1_keys[:, 1], repeats_for_d1)
        total_pairs = np.sum(c1.astype(np.int64) * c2)
        if total_pairs == 0:
            self.calculate_scores(0, 0)
            return

        start2 = np.insert(np.cumsum(c2), 0, 0)[:-1]
        base = np.repeat(np.repeat(start2, c1), repeats_for_d1)

        offsets = np.ones(total_pairs, dtype=int)
        reset_indices = np.cumsum(repeats_for_d1)[:-1]

        if reset_indices.size > 0:
            offsets[reset_indices] = 1 -repeats_for_d1[:-1]

        offsets[0] = 0
        offsets = np.cumsum(offsets)

        e2_pairs = entity_2_keys[base + offsets, 1]

        # 7. --- MEMORY-EFFICIENT DISTINCT PAIRS ---
        # Stack into a contiguous 2D array
        pairs = np.empty((total_pairs, 2), dtype=blocks_with_keys.dtype)
        pairs[:, 0] = e1_pairs
        pairs[:, 1] = e2_pairs

        # Void view trick: Treat rows as single byte-strings.
        # This bypasses the heavy memory overhead of np.unique(..., axis=0)
        void_dt = np.dtype((np.void, pairs.dtype.itemsize * 2))
        void_view = pairs.view(void_dt)


        unique_candidates_void = np.unique(void_view)

        true_positives = 0
        total_matching_blocks = np.unique(void_view).shape[0]

        gt_array = np.ascontiguousarray(self.encoded_data.ground_truth.values, dtype=pairs.dtype)
        gt_array[:, 1] += self.encoded_data.bounds[0]
        gt_void_view = gt_array.view(void_dt)

        true_positives = np.intersect1d(unique_candidates_void, gt_void_view, assume_unique=True).shape[0]



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
