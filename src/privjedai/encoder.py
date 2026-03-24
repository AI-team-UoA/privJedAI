"""PPRL Encoders Class"""
from dataclasses import dataclass
from typing import List, Literal, Dict, Set, Optional, Union
import re
import hashlib
import pandas as pd
from metaphone import doublemetaphone
from tqdm.auto import tqdm
from privjedai.encoded_data import BloomEncodedData

@dataclass
class BloomFilterConfig:
    """Configuration class for Bloom Filters"""
    size: int = 1024
    num_hashes: int = 20
    offset: int = 0
    hashing_type : Literal["salted_string", "salted_qgrams",
            "salted_skipqgrams", "salted_metaphone",
            "salted_tokens"] = "salted_string"
    qgrams : int = 5
    salt : str = ""
    attributes : Optional[List[str]] = None


class BloomFilter:
    """
    Bloom filter encoder for privacy-preserving record linkage.

    Creates Bloom filter representations of data using various hashing techniques
    to enable privacy-preserving similarity comparisons.
    """

    VALID_HASHING_TYPES = [
        "salted_string","salted_qgrams",
        "salted_skipqgrams", "salted_metaphone",
        "salted_tokens"
    ]


    TOKEN_SPLIT_RE = re.compile(r'[\W_]+')

    def __init__(self, config: BloomFilterConfig):
        """
        Initialize Bloom filter encoder.

        Args:
            size (int): Bloom filter size in bits
            num_hashes (int): Number of hash functions to use
            offset (int): Starting offset for bit positions
            hashing_type (literal): Method for encoding strings
            qgrams: Q-gram size for qgram-based methods
            pad: Whether to pad short qgrams
            salt: Salt for hashing
            attributes: Specific attributes to encode
        """



        self.size : int  = config.size
        self.num_hashes : int = config.num_hashes
        self.offset : int = config.offset
        self.qgrams = config.qgrams
        self.salt = config.salt
        self.attributes = config.attributes
        self.hashing_type = config.hashing_type



    def generate_hash(self, str_to_enc: str) -> List[int]:
        """Return Bloom filter bit positions
        for the given string using the selected hashing method.

        Args:
            str_to_enc (str): String to encode

        Returns:
            List[int]: List of the positive bits
        """
        encoder_dict = {
            "salted_string" : lambda: self._salted_string(str_to_enc),
            "salted_qgrams" : lambda: self._salted_qgrams(str_to_enc),
            "salted_skipqgrams" : lambda: self._salted_skipqgrams(str_to_enc),
            "salted_metaphone" : lambda: self._salted_metaphone(str_to_enc),
            "salted_tokens" : lambda: self._salted_token(str_to_enc),
        }
        return encoder_dict[self.hashing_type]()

    def _create_bloom_dict_for_row(self, row : tuple) -> Dict[str, List[int]]:
        bloom_dict = {}
        if self.attributes:
            for attribute in self.attributes:
                bloom_dict[attribute] = self.generate_hash(getattr(row,attribute))
        else:
            flatten_row = " ".join(str(x) for x in row)
            bloom_dict['row'] = self.generate_hash(flatten_row)
        return bloom_dict


    def _encode(self, df: pd.DataFrame) -> BloomEncodedData:
        bloom_dict = {}
        total_length = len(df)
        for i, row in tqdm(enumerate(df.itertuples(index=False)),
                        total=total_length,
                        desc=f'Encoding Data with attributes {self.attributes}',
                        leave=False, position=1):
            bloom_dict[i] = self._create_bloom_dict_for_row(row)
        return BloomEncodedData(data = bloom_dict, length=self.size + self.offset, )  #leng

    def _hash_token(self, bf: Set[int],  token: Union[str, bytes], hash_token : bool = True):

        if hash_token:
            token_bytes: bytes = token if isinstance(token, bytes) else b'0'
            hashed = hashlib.sha256(token_bytes).digest()
            for i in range(self.num_hashes):
                start_idx = i * 4 % (len(hashed) - 4)
                segment = hashed[start_idx:start_idx + 4]
                position = (int.from_bytes(segment, 'little') % self.size) + self.offset
                bf.add(position)
        else:
            if len(token) < self.qgrams:
                token = token.ljust(self.qgrams, '_')

            for i in range(len(token) - self.qgrams + 1):
                qgram_token = token[i:i+self.qgrams]
                base_string = f"{qgram_token}{self.salt}".encode("utf-8")
                hashed = hashlib.sha256(base_string).digest()

                h1 = int.from_bytes(hashed[:16], 'little')
                h2 = int.from_bytes(hashed[16:], 'little')

                for j in range(self.num_hashes):
                    combined_hash = (h1 + j * h2) % self.size
                    position = combined_hash + self.offset
                    bf.add(position)
                # for j in range(self.num_hashes):
                #     salted_token = f"{qgram_token}{j}{self.salt}".encode("utf-8")
                #     hashed = hashlib.sha256(salted_token).digest()
                #     position = (int.from_bytes(hashed, "little") % self.size) + self.offset
                #     bf.add(position)

        return bf


    def _salted_string(self, str_to_enc: str) -> List[int]:
        bf = set()
        for i in range(self.num_hashes):
            salted_string = (str_to_enc + str(i) +  self.salt).encode("utf-8")
            bf = self._hash_token(bf, salted_string, True)

        return [*bf]

    def _salted_qgrams(self, str_to_enc: str) -> List[int]:
        tokens : List[str] = self._tokenize(str_to_enc)
        split = [f"_{word}_" for word in tokens if word]
        bf = set()
        for token in split:
            bf = self._hash_token(bf, token, False)                   # HASH QGRAM

        return [*bf]

    def _salted_skipqgrams(self, str_to_enc: str) -> List[int]:
        """" Gets two different strings
        one without the chars in the odd positions
        and one without the even ones.
        If the string is shorter of qgrams,
        # is added for creating the qgrams """

        bf = set()

        tokens = self._tokenize(str_to_enc)
        split = [f"_{word}_" for word in tokens if word]

        for token in split:
            skipped_tokens : List[str] = [token[1::2], token[::2]]
            for skipped_token in skipped_tokens:
                bf = self._hash_token(bf, skipped_token, False)                   # HASH QGRAM

        return [*bf]

    def _salted_metaphone(self, str_to_enc: str) -> List[int]:
        """" Hashes metaphone if it exists else hashes token """
        bf = set()
        # bf = Bloom(capacity=self.size, error_rate=self.error_rate)
        tokens = self._tokenize(str_to_enc)
        for token in tokens:
            for i in range(self.num_hashes):
                salted_token : str = self.salt + str(i) + token
                double_metaphone = doublemetaphone(salted_token)

                if len(double_metaphone[0]) == 0 and len(double_metaphone[1])  == 0:
                    utf_8_token = salted_token.encode('utf-8')
                    bf = self._hash_token(bf, utf_8_token, False)                   # HASH QGRAM
                    continue

                for metaphone in double_metaphone:
                    if '' == metaphone:
                        continue
                    hashed = hashlib.sha256(metaphone.lower().encode('utf-8')).digest()
                    digest_as_int = (int.from_bytes(hashed, 'little') % self.size )  + self.offset
                    bf.add(digest_as_int)

        return [*bf]


    def _salted_token(self, str_to_enc: str) -> List[int]:
        """" Hashes tokens and extract from the hashed the qgrams """
        bf = set()
        tokens = self._tokenize(str_to_enc)
        split = [f"_{word}_" for word in tokens if word]
        for token in split:
            for i in range(self.num_hashes):
                salted_token = (token + str(i) +  self.salt).encode('utf-8')
                bf = self._hash_token(bf, salted_token, True)

        return [*bf]


    def  _tokenize(self, str_to_enc : str) -> List:
        return list(set(filter(None, self.TOKEN_SPLIT_RE.split(str_to_enc.lower()))))


    def encode(self, df: pd.DataFrame) -> BloomEncodedData:
        """Encoding Data to Bloom Filters.
        Attributes must be the same with the other datasets,
        that are gonna be matched.

        Args:
            df (pd.DataFrame): Dataset to be encoded

        Returns:
            BloomEncodedData: An object with bloomfilters of your data
        """
        columns = list(df.columns)
        if self.attributes:
            for col in self.attributes:
                if col not in columns:
                    raise ValueError("Attributes must be the same between the two dataframes")
        else:
            self.attributes = columns

        df = df[self.attributes] if self.attributes else df
        return self._encode(df)



# class HomomorphicEncryption:
#     def __init__(self,
#         qgrams : int = 2,
#         pad : bool = True,
#         salt : str = "",
#         attributes : List[str] = None,
#     ):
#         self.qgrams : int = qgrams
#         self.pad : bool = pad
#         self.salt : str = salt
#         self.attributes : List[str] = attributes

#     def _hash(self, str_to_enc: str) -> List[int]:
#         tokens = list(set(filter(None, re.split('[\\W_]', str_to_enc.lower()))))
#         qgrams_set = set()
#         for token in tokens:
#             salted_token = self.salt + token
#             # if len(salted_token) < self.qgrams and self.pad:
#             #     salted_token = salted_token + '#'*(self.qgrams - len(salted_token))

#             for i in range(len(salted_token) - self.qgrams + 1):
#                 qgram_token = str(salted_token[i: i+self.qgrams])
#                 if len(qgram_token) < self.qgrams and self.pad:
#                     qgram_token = qgram_token + '#'*(self.qgrams - len(qgram_token))
#                 qgrams_set.add(qgram_token)
#         vector_uint64 : List[int] = []
#         for token in list(qgrams_set):
#             uint64 : int = 0
#             for idx, c in enumerate(reversed(token)):
#                 uint64 += (ord(c) & 0xff) << (8 * idx)
#             vector_uint64.append(uint64)

#         return vector_uint64

#     def _create_vector_for_row(self, row : pd.Series) -> Dict[str, List[int]]:
#         homomorphic_vector = dict()
#         if self.attributes:
#             for attribute in self.attributes:
#                 vector = self._hash(str(row[attribute]))
#                 if len(vector) > 0:
#                     homomorphic_vector[attribute] = vector

#         else:
#             flatten_row : list = row.values.tolist()
#             flatten_row = " ".join(str(x) for x in flatten_row)
#             homomorphic_vector['row'] = self._hash(flatten_row)
#         return homomorphic_vector

#     def _encode(self) -> HomomorphicEcnodedData:
#         df_1 : pd.DataFrame =
# self.data.dataset_1[self.attributes] if self.attributes else self.data.dataset_1
#         df_2 : pd.DataFrame =
#  self.data.dataset_2[self.attributes] if self.attributes else self.data.dataset_2
#         homomorphic_vector = dict()
#         for id, (_, row) in enumerate(df_1.iterrows()):
#             homomorphic_vector[id] = self._create_vector_for_row(row)
#         for id, (_, row) in enumerate(df_2.iterrows(), start = self.data.dataset_limit):
#             homomorphic_vector[id] = self._create_vector_for_row(row)

#         return HomomorphicEcnodedData(data=homomorphic_vector)

#     def encode(self, data: Data) -> HomomorphicEcnodedData :
#         self.data = data
#         columns_1 = list(data.dataset_1.columns)
#         columns_2 = list(data.dataset_2.columns)

#         if self.attributes:
#             for col in self.attributes:
#                 if col not in columns_1 or col not in columns_2:
#                     raise ValueError("Attributes must be the same between the two dataframes")
#         else:
#             upperset : set =
# set(columns_1) if len(set(columns_1)) > len(set(columns_2)) else set(columns_2)
#             subset : set =
# set(columns_1) if len(set(columns_1)) <= len(set(columns_2)) else set(columns_2)
#             self.attributes = list(subset) if subset.issubset(upperset) else None
#         print(f'Encrypting these attributes: {self.attributes}')

#         return self._encode()
