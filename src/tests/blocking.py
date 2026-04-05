"""Tests for blocking.py"""

from unittest.mock import MagicMock

from bitarray import bitarray
import pytest

# Assuming your classes are accessible, e.g., from your_module
from privjedai.blocking import LSHBlocker, BitBlocker, FAISSBlocking

# If they are in the same file, you can import them directly.


def create_mock_encoded_data(attributes= None, length=12):
    """ --- Helper function to create the mock data structure --- """

    mock_data = MagicMock()
    attributes = ['a', 'b'] if not attributes else attributes
    mock_data.metadata.attributes = attributes
    mock_data.metadata.length = length
    # Simulate two datasets: records 0-2 (DS1) and 3-4 (DS2)
    mock_data.bounds = [3, 6]

    # Create dummy bitarray_dict
    bitarray_dict = {}
    bfs = ['0000', '0001', '0010']

    for i in range(3):
        # Create a simple, deterministic bitarray for testing
        # e.g., all 0s for index < 3, all 1s for index >= 3
        dummy_bf = bitarray(bfs[i] * int(length / 4))
        bitarray_dict[i] = dict.fromkeys(attributes, dummy_bf)
        bitarray_dict[i+3] =  dict.fromkeys(attributes, dummy_bf)



    mock_data.bitarray_dict = bitarray_dict

    # Mock ground truth for evaluation tests
    mock_data.skip_ground_truth = False
    mock_data.ground_truth = MagicMock() # Mock the dataframe/iterator

    # For FAISSBlocking evaluate
    mock_data.ground_truth.iterrows.return_value = [
        (0, 0,0),
        (1, 1,1),
        (2, 2,2)
    ]

    mock_data.ground_truth.itertuples.return_value = [
        (0,0,0),
        (1,1,1),
        (2,2,2)
    ]
    mock_data.ground_truth.values = [
        (0, 0),
        (1, 1),
        (2, 2)
    ]

    mock_data.ground_truth.__len__.return_value = 3 # Or any other iterable with 3 elements

    return mock_data




# --- Test class for LSHBlocker ---
class TestLSHBlocker:
    """Test lsh blocker"""

    def test_set_attributes(self):
        blocker = LSHBlocker(psi=4, lambda_=2, seed=1, prune_ratio=0.9)
        mock_data = create_mock_encoded_data( attributes=['a', 'b'])

        with pytest.raises(ValueError):
            _ = blocker.build_blocks(mock_data, attributes=['a', 'b', 'c'])
        with pytest.raises(ValueError):
            _ = LSHBlocker(psi=0, lambda_=0, seed=1, prune_ratio=0.9)


    def test_build_blocks(self):
        blocker = LSHBlocker(psi=4, lambda_=2, seed=1, prune_ratio=0.9)
        blocker.blocks = None
        with pytest.raises(AttributeError):
            blocker.evaluate({})
        mock_data = create_mock_encoded_data( attributes=['a', 'b'])

        _ = blocker.build_blocks(mock_data, attributes=['a', 'b'])
        blocker.evaluate({})

        blocker.encoded_data = None
        with pytest.raises(AttributeError):
            blocker.evaluate({})
            blocker.report()

        blocker.encoded_data = mock_data
        blocker.encoded_data.skip_ground_truth = True
        with pytest.raises(AttributeError):
            blocker.evaluate({})





    def test_lsh_build_blocks_flow(self):
        """Test for lsh"""

        blocker = LSHBlocker(psi=4, lambda_=2, seed=1, prune_ratio=0.9)
        mock_data = create_mock_encoded_data( attributes=['a', 'b'])

        with pytest.raises(AttributeError):
            blocker.report()
        conf = blocker._configuration()

        assert conf["psi"] == 4

        with pytest.raises(ValueError):
            _ = blocker._select_bit_positions([bitarray('0100')])

        blocks = blocker.build_blocks(mock_data, attributes=['a', 'b'])
        blocker.report()

        assert isinstance(blocks, dict)


# --- Test class for BitBlocker ---
class TestBitBlocker:
    """TestBitBlocker"""


    def test_args(self):
        """Check if wrong arguments raise Errors"""
        with pytest.raises(ValueError):
            _ = BitBlocker(psi=0, lambda_=0, seed=1)
        blocker = BitBlocker(psi=4, lambda_=2, seed=1)
        mock_data = create_mock_encoded_data(attributes=['a', 'b'], length=5)

        # ACT
        with pytest.raises(ValueError):
            blocker.build_blocks(mock_data, attributes=['a', 'b'])



    def test_bitblocker_fit(self):
        """Bit Blocker fit"""
        # ARRANGE
        blocker = BitBlocker(psi=4, lambda_=2, seed=1)
        mock_data = create_mock_encoded_data(attributes=['a', 'b'], length=12)

        # ACT
        blocker.build_blocks(mock_data, attributes=['a', 'b'])

        # ASSERT
        # Check if hash_len is correct (10 bits * 2 attributes = 20)
        assert blocker.hash_len == 24
        # Check if hash_indices has the right dimensions (lmbda x psi)
        assert len(blocker.hash_indices) == 2
        assert len(blocker.hash_indices[0]) == 4
        # Check that indices are within the correct range [0, 20)
        assert all(0 <= idx < 24 for indices in blocker.hash_indices for idx in indices)

    def test_evaluate(self):
        """Evaluation testing"""
        mock_data = create_mock_encoded_data(attributes=['a', 'b'], length=12)
        predicted = {0 : {3}, 1 : {4}, 2 : {5}}
        blocker = BitBlocker(psi=4, lambda_=2, seed=1)
        _ = blocker.build_blocks(mock_data)
        ev = blocker.evaluate(predicted)


        assert int(ev['F1 %']) == 100


class TestFAISS:
    """"TEST FAISS"""

    def test_faiss(self):
        """"TEST FAISS"""
        #  ARRANGE
        blocker = FAISSBlocking()

        mock_data = create_mock_encoded_data(length=24, attributes=['a', 'b'])

        _ = blocker.build_blocks(encoded_data=mock_data, attributes=['a', 'b'], top_k=100)

        # ACT
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['a', 'b'], top_k=1)

        assert blocks[0] == {3}
        mock_data = create_mock_encoded_data(length=24, attributes=['b'])
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['b'], top_k=1)

        assert blocks[0] == {3}

        ev = blocker.evaluate(blocks)
        print(blocks)
        print(ev)
        assert int(ev['F1 %']) == 100

        blocker = FAISSBlocking('hnsw')
        mock_data = create_mock_encoded_data(length=24, attributes=['a', 'b'])
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['a', 'b'], top_k=100)
        ev = blocker.evaluate(blocks)
        assert int(ev['Recall %']) == 100


        with pytest.raises(AttributeError):
            blocker.encoded_data.skip_ground_truth = True
            blocker.evaluate(blocks)

        with pytest.raises(AttributeError):
            blocker.encoded_data = None
            blocker.evaluate(blocks)

        blocker = FAISSBlocking('lsh')
        mock_data = create_mock_encoded_data(length=24, attributes=['a', 'b'])
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['a', 'b'], top_k=100)
        ev = blocker.evaluate(blocks)
        assert int(ev['Recall %']) == 100

        with pytest.raises(AttributeError):
            blocker.encoded_data.skip_ground_truth = True
            blocker.evaluate(blocks)

        with pytest.raises(AttributeError):
            blocker.encoded_data = None
            blocker.evaluate(blocks)

