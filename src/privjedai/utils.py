"""Utils methods for many classes"""
from typing import  Dict
import numpy as np
from privjedai.datamodel import Block

POPCOUNT_TABLE = np.array([bin(x).count('1') for x in range(256)], dtype=np.uint8)


def chi_square(in_array: np.array) -> float:
    """Chi Square Method

    Args:
        in_array (np.array): Input array

    Returns:
        float: Statistic computation of Chi Square.
    """
    row_sum, column_sum, total = \
        np.sum(in_array, axis=1), np.sum(in_array, axis=0), np.sum(in_array)
    sum_sq = expected = 0.0
    for r in range(0, in_array.shape[0]):
        for c in range(0, in_array.shape[1]):
            expected = (row_sum[r]*column_sum[c])/total
            sum_sq += ((in_array[r][c]-expected)**2)/expected
    return sum_sq


def get_blocks_cardinality(blocks: Dict[str, Block]) -> int:
    """Returns the cardinality of the blocks.

    Args:
        blocks (dict): Blocks.

    Returns:
        int: Cardinality.
    """
    return sum(block.get_cardinality() for block in blocks.values())

def are_matching(entity_index : Dict[int, set], id1, id2) -> bool:
    '''
    id1 and id2 consist a matching pair if:
    - Blocks: intersection > 0 (comparison of sets)
    - Clusters: cluster-id-j == cluster-id-i (comparison of integers)
    '''

    if len(entity_index) < 1:
        raise ValueError("No entities found in the provided index")
    if isinstance(list(entity_index.values())[0], set): # Blocks case
        return not entity_index[id1].isdisjoint(entity_index[id2])
    return entity_index[id1] == entity_index[id2] # Clusters case

def batch_pairs(iterable, batch_size: int = 1):
    """
    Generator function that breaks an iterable into batches of a set size.
    :param iterable: The iterable to be batched.
    :param batch_size: The size of each batch.
    """
    return (iterable[pos:pos + batch_size] for pos in range(0, len(iterable), batch_size))


# Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
def _drop_single_entity_blocks(blocks : dict[Block]) -> dict:
    return dict(filter(lambda e: not _block_with_one_entity(e[1]), blocks.items()))

def _block_with_one_entity(block : Block) -> bool:
    return len(block.entities) == 1

def _math_dice(intersecntion: int, cadinality_a : int, cardinality_b : int) -> float:
    return float(2 * (intersecntion) / (cadinality_a + cardinality_b))



def _tversky(blooms_1: np.array, blooms_2: np.array, alpha: float = 1, beta: float =  1) -> np.array:

    intersection_bytes = np.bitwise_and(blooms_1, blooms_2)
    intersections = np.bitwise_count(intersection_bytes).sum(axis=2)

    card_1 = np.sum(blooms_1, axis=2)
    card_2 = np.sum(blooms_2, axis=2)

    # Tversky index: |A ∩ B| / (|A ∩ B| + α * |A \ B| + β * |B \ A|)
    a_minus_b = card_1 - intersections
    b_minus_a = card_2 - intersections
    denominators = intersections + alpha * a_minus_b + beta * b_minus_a

    scores = np.zeros_like(intersections, dtype=np.float64)
    valid_mask = denominators > 0
    scores[valid_mask] = intersections[valid_mask] / denominators[valid_mask]
    avg_similarities = np.mean(scores, axis=1)

    return avg_similarities

def _dice(blooms_1 : np.array, blooms_2 : np.array) -> np.array:

    intersection_bytes = np.bitwise_and(blooms_1, blooms_2)
    intersections = intersection_bytes.sum(axis=2)

    card_1 = np.sum(blooms_1, axis=2)  # (n_candidates, n_attributes)
    card_2 = np.sum(blooms_2, axis=2)  # (n_candidates, n_attributes)

    denominators = card_1 + card_2
    dice_scores = np.zeros_like(intersections, dtype=np.float64)
    valid_mask = denominators > 0
    dice_scores[valid_mask] = (2 * intersections[valid_mask]) / denominators[valid_mask]
    avg_similarities = np.mean(dice_scores, axis=1)

    return avg_similarities

def _jaccard(blooms_1: np.array, blooms_2: np.array):

    intersection_bytes = np.bitwise_and(blooms_1, blooms_2)
    intersections = intersection_bytes.sum(axis=2)

    # 2. Cardinalities (Count of True)
    card_1 = np.sum(blooms_1, axis=2)
    card_2 = np.sum(blooms_2, axis=2)

    union = card_1 + card_2 - intersections
    scores = np.zeros_like(intersections, dtype=np.float64)
    mask = union > 0
    scores[mask] = intersections[mask] / union[mask]
    avg_similarities = np.mean(scores, axis=1)
    return avg_similarities

def _cosine(blooms_1: np.array, blooms_2: np.array):

    intersection_bytes = np.bitwise_and(blooms_1, blooms_2)
    intersections = intersection_bytes.sum(axis=2)

    # 2. Cardinalities (Count of True)
    card_1 = np.sum(blooms_1, axis=2)
    card_2 = np.sum(blooms_2, axis=2)

    denom = np.sqrt(card_1 * card_2)
    scores = np.zeros_like(intersections, dtype=np.float64)
    mask = denom > 0
    scores[mask] = intersections[mask] / denom[mask]
    avg_similarities = np.mean(scores, axis=1)
    return avg_similarities


def _scm(blooms_1 : np.array, blooms_2 : np.array) -> np.array:

    length = blooms_1.shape[2]
    xor_bytes = np.bitwise_xor(blooms_1, blooms_2)
    xor_sum = xor_bytes.sum(axis=2)
    matches = length - xor_sum
    scm_per_attr = matches / length
    avg_similarities = np.mean(scm_per_attr, axis=1)

    return avg_similarities



