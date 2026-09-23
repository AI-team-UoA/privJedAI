"""Evaluation module
This file contains all the methods for evaluating every module in pyjedai.
"""
from collections import defaultdict
from numba import b1
import numpy as np
from scipy.sparse import csr_matrix
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# 1. Filter into D1 and D2
        mask = blocks_with_keys[:, 1] < limit_

        # np.unique with axis=0 removes duplicate (block, entity) entries early
        b1_keys, e1_keys = np.unique(blocks_with_keys[mask], axis=0).T
        b2_keys, e2_keys = np.unique(blocks_with_keys[~mask], axis=0).T

        if b1_keys.size == 0 or b2_keys.size == 0:
            return self.calculate_scores(0, 0)

        # 2. Keep only common blocks to aggressively reduce matrix size
        common_blocks = np.intersect1d(b1_keys, b2_keys, assume_unique=False)
        if common_blocks.size == 0:
            return self.calculate_scores(0, 0)

        m1 = np.isin(b1_keys, common_blocks)
        b1_keys, e1_keys = b1_keys[m1], e1_keys[m1]

        m2 = np.isin(b2_keys, common_blocks)
        b2_keys, e2_keys = b2_keys[m2], e2_keys[m2]

        del mask, blocks_with_keys, m1, m2
        gc.collect()

        # 3. Map Blocks and Entities to continuous 0-indexed integers
        unique_blocks, b_idx1 = np.unique(b1_keys, return_inverse=True)
        # b2_keys are guaranteed to be in unique_blocks, so searchsorted is safe & fast
        b_idx2 = np.searchsorted(unique_blocks, b2_keys)
        num_blocks = unique_blocks.size

        u_e1, e_idx1 = np.unique(e1_keys, return_inverse=True)
        u_e2, e_idx2 = np.unique(e2_keys, return_inverse=True)

        # 4. --- GRAPH MATRIX CREATION ---
        # Matrix A: Blocks -> D1 Entities (shape: num_blocks x unique_D1)
        # Matrix B: Blocks -> D2 Entities (shape: num_blocks x unique_D2)
        A = csr_matrix((np.ones(len(b_idx1), dtype=np.int8), (b_idx1, e_idx1)), shape=(num_blocks, len(u_e1)))
        B = csr_matrix((np.ones(len(b_idx2), dtype=np.int8), (b_idx2, e_idx2)), shape=(num_blocks, len(u_e2)))

        del b1_keys, e1_keys, b2_keys, e2_keys, b_idx1, b_idx2
        gc.collect()

        # 5. --- BOTTLE-NECK DESTROYER ---
        # Multiply transposed A by B.
        # C[i, j] will hold the number of blocks shared by Entity1_i and Entity2_j
        C = A.T @ B

        # Any non-zero element in C represents a distinct candidate pair.
        # This completely bypasses deduplication because sparse matrices naturally only store non-zeros.
        total_matching_blocks = C.nnz

        if total_matching_blocks == 0:
            return self.calculate_scores(0, 0)

        # 6. --- TRUE POSITIVES (Fast Matrix Intersection) ---
        gt = self.encoded_data.ground_truth.values
        gt_e1 = gt[:, 0]
        gt_e2 = gt[:, 1] + self.encoded_data.bounds[0]

        # Filter GT to only entities that exist in our candidate matrices
        valid_mask = np.isin(gt_e1, u_e1) & np.isin(gt_e2, u_e2)
        gt_e1_valid = gt_e1[valid_mask]
        gt_e2_valid = gt_e2[valid_mask]

        if gt_e1_valid.size > 0:
            # Map valid GT IDs to the sparse matrix indices
            gt_idx1 = np.searchsorted(u_e1, gt_e1_valid)
            gt_idx2 = np.searchsorted(u_e2, gt_e2_valid)

            # Create a sparse matrix representing the Ground Truth pairs
            GT_sparse = csr_matrix((np.ones(len(gt_idx1), dtype=np.int8), (gt_idx1, gt_idx2)), shape=C.shape)

            # Element-wise multiply our Candidates (C) by the Ground Truth.
            # Non-zeros will only survive if they exist in BOTH matrices!
            intersection = C.multiply(GT_sparse)
            true_positives = intersection.nnz
        else:
            true_positives = 0

        del A, B, C, u_e1, u_e2, GT_sparse, intersection
        gc.collect()

        return self.calculate_scores(true_positives, total_matching_blocks)
        # repeats_for_d1 = np.repeat(c2, c1)
        # e1_pairs = np.repeat(entity_1_keys[:, 1], repeats_for_d1)


        # total_pairs = np.sum(c1.astype(np.int64) * c2)
        # if total_pairs == 0:
        #     return self.calculate_scores(0, 0)

        # start2 = np.insert(np.cumsum(c2), 0, 0)[:-1]
        # base = np.repeat(np.repeat(start2, c1), repeats_for_d1)

        # offsets = np.ones(total_pairs, dtype=int)
        # reset_indices = np.cumsum(repeats_for_d1)[:-1]

        # if reset_indices.size > 0:
        #     offsets[reset_indices] = 1 - repeats_for_d1[:-1]

        # offsets[0] = 0
        # offsets = np.cumsum(offsets)

        # # Fetch the corresponding D2 elements
        # e2_pairs = entity_2_keys[base + offsets, 1]

        # # 7. --- MEMORY-EFFICIENT DISTINCT PAIRS ---
        # # Stack into a contiguous 2D array
        # pairs = np.empty((total_pairs, 2), dtype=blocks_with_keys.dtype) # Use your original array's dtype
        # pairs[:, 0] = e1_pairs
        # pairs[:, 1] = e2_pairs

        # # Void view trick: Treat rows as single byte-strings.
        # # This bypasses the heavy memory overhead of np.unique(..., axis=0)
        # void_dt = np.dtype((np.void, pairs.dtype.itemsize * 2))
        # void_view = pairs.view(void_dt)

        # unique_candidates_void = np.unique(void_view)
        # total_matching_blocks = unique_candidates_void.shape[0]


        # gt_array = np.ascontiguousarray(self.encoded_data.ground_truth.values, dtype=pairs.dtype)
        # gt_array[:, 1] += self.encoded_data.bounds[0]
        # gt_void_view = gt_array.view(void_dt)

        # true_positives = np.intersect1d(unique_candidates_void, gt_void_view, assume_unique=True).shape[0]

        # del pairs, void_view, unique_candidates_void, gt_void_view, gt_array
        # del e1_pairs, e2_pairs, base, offsets, repeats_for_d1, reset_indices
        # gc.collect()


        # return self.calculate_scores(true_positives, total_matching_blocks)
        # d1_groups = np.split(entity_1_keys[:, 1], np.cumsum(c1)[:-1])
        # d2_groups = np.split(entity_2_keys[:, 1], np.cumsum(c2)[:-1])

        # total_d1_entities = np.unique(entity_1_keys[:,1]).size
        # total_d2_entities = np.unique(entity_2_keys[:,1]).size

        # max_possible_pairs = total_d1_entities * total_d2_entities

        # # 6. --- CHUNKED CARTESIAN PRODUCT & AGGREGATION ---
        # unique_candidate_pairs = np.array([], dtype=np.uint64)
        # pair_buffer = []
        # buffer_size = 0
        # FLUSH_LIMIT = 10_000_000  # Flush memory every 10M pairs

        # for g1, g2 in zip(d1_groups, d2_groups):
        #     # Create a grid for the current block
        #     e1_grid, e2_grid = np.meshgrid(g1, g2, indexing='ij')

        #     # Pack (e1, e2) into a single uint64 integer: (e1 << 32) | e2
        #     packed_pairs = (e1_grid.ravel().astype(np.uint64) << 32) | e2_grid.ravel().astype(np.uint64)

        #     pair_buffer.append(packed_pairs)
        #     buffer_size += packed_pairs.size

        #     # Periodically deduplicate to prevent memory explosions
        #     if buffer_size >= FLUSH_LIMIT:
        #         batch = np.concatenate(pair_buffer)
        #         unique_candidate_pairs = np.unique(np.concatenate([unique_candidate_pairs, batch]))
        #         pair_buffer = []
        #         buffer_size = 0

        #         if unique_candidate_pairs.size == max_possible_pairs:
        #             return self.calculate_scores(self.encoded_data.ground_truth.values.shape[0],
        #                                          max_possible_pairs)

        # # Flush remaining items in the buffer
        # if pair_buffer:
        #     batch = np.concatenate(pair_buffer)
        #     unique_candidate_pairs = np.unique(np.concatenate([unique_candidate_pairs, batch]))

        # total_matching_blocks = unique_candidate_pairs.size

        # if total_matching_blocks == max_possible_pairs:
        #     return self.calculate_scores(self.encoded_data.ground_truth.values.shape[0],
        #                                  total_matching_blocks)


        # # 7. --- TRUE POSITIVES CALCULATION ---
        # # Pack the ground truth values using the same bitshift logic
        # gt_array = np.ascontiguousarray(self.encoded_data.ground_truth.values)
        # gt_array[:, 1] += self.encoded_data.bounds[0]
        # gt_packed = (gt_array[:, 0].astype(np.uint64) << 32) | gt_array[:, 1].astype(np.uint64)

        # # Intersect the two 1D uint64 arrays
        # true_positives = np.intersect1d(unique_candidate_pairs, gt_packed, assume_unique=True).size

        # return self.calculate_scores(true_positives, total_matching_blocks)


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

        del unique_keys, inverse_keys, entity_1_keys, entity_2_keys, cnt_entity_1_blocks, cnt_entity_2_blocks, blocks_cardinalities
        del id_to_keys
        gc.collect()

        self.calculate_scores(true_positives, total_matching_blocks)
