"""Schemas for privJedAI API"""

from typing import List, Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field
from fastapi import UploadFile, File


class SchemaBloomFilterConfig(BaseModel):
    """Schema to encode to BloomFilter"""

    file: UploadFile = File(..., description="File to encode to BloomFilter")
    separator: str = Field(
        ..., description="Separator used in the file to encode to BloomFilter"
    )
    bloom_filter_size: int = Field(..., description="Size of the BloomFilter")
    number_of_hashes: int = Field(
        ..., description="Number of hashes used in the BloomFilter"
    )
    hashing_type: Literal[
        "salted_string",
        "salted_qgrams",
        "salted_skipqgrams",
        "salted_metaphone",
        "salted_tokens",
    ] = Field("salted_string", description="Type of hashing used in the BloomFilter")
    qgrams: Optional[int] = Field(
        2, ge=1, le=6, description="Number of q-grams used in the BloomFilter"
    )
    salt: Optional[str] = Field("", description="Salt used in the hashing")
    attributes: Optional[List[str]] = Field(
        None, description="List of attributes to encode to BloomFilter"
    )


class LSHBlockingAuto(BaseModel):
    """LSHBlockingAuto schema"""

    type: Literal["auto"]
    auto_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Auto threshold for LSH blocking, \
                expressed as a percentage of Hamming Distance Threshold (max allowed differences). \
                This parameter is used if psi lambda is not provided."
    )


class LSHBlockingManual(BaseModel):
    """Schema for LSH blocking method"""

    type: Literal["manual"]

    psi: Optional[int] = Field(20, ge=1, le=2000)
    lambda_: int = Field(10, ge=1, le=150)


class SchemaBlockFiltering(BaseModel):
    """Schema for block filtering method"""

    type: Literal["block_filtering"]
    ratio: float = Field(0.8, ge=0.0, le=1.0, description="Ratio for block filtering, \
        expressed as a percentage of the total number of blocks")


class SchemaBlockPurging(BaseModel):
    """Schema for block purging method"""

    type: Literal["block_purging"]

    smoothing_factor: float = Field(
        1.025, ge=0.0, le=2.0, description="Ratio for block purging, \
        expressed as a percentage of the total number of blocks"
    )


class Combination(BaseModel):
    """Schema for combination of block filtering and block purging methods"""

    type: Literal["combination"]

    ratio: float = Field(0.8, ge=0.0, le=1.0, description="Ratio for block filtering, \
        expressed as a percentage of the total number of blocks")

    smoothing_factor: float = Field(
        1.025, ge=0.0, le=2.0, description="Ratio for block purging, \
        expressed as a percentage of the total number of blocks"
    )


class Metablocking(BaseModel):
    """Schema for metablocking method"""

    method: Literal[
        "Weighted Edge Pruning",
        "Weighted Node Pruning",
        "Cardinality Edge Pruning",
        "Cardinality Node Pruning",
        "BLAST",
    ] = "Weighted Edge Pruning"

    weighting_scheme: Literal[
        "CN-CBS",
        "CBS",
        "SN-CBS",
        "SNC",
        "CNC",
        "SND",
        "CND",
        "CNJ",
        "SNJ",
        "COSINE",
        "DICE",
        "ECBS",
        "JS",
        "EJS",
        "X2",
    ] = "CN-CBS"


class Matching(BaseModel):
    """Schema for matching method"""

    batch_size: Optional[int] = Field(
        10_000, ge=1, le=1_000_000, description="Batch size for matching"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Threshold for matching, \
        expressed as a percentage of similarity score")
    metric: Literal["dice", "scm", "jaccard", "cosine"] = "dice"


class Clustering(BaseModel):
    """Schema for clustering method"""

    method: Literal[
        "Connected Components Clustering",
        "Unique Mapping Clustering",
        "Kiraly Clustering",
    ] = "Connected Components Clustering"
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Threshold for clustering, \
        expressed as a percentage of similarity score"
    )


class LSHWorkflow(BaseModel):
    """Schema for LSH workflow"""

    type: Literal["lsh"]
    blocking: Annotated[
        Union[LSHBlockingAuto, LSHBlockingManual], Field(discriminator="type")
    ]
    filtering_method: Optional[
        Annotated[
            Union[SchemaBlockFiltering, SchemaBlockPurging, Combination],
            Field(discriminator="type"),
        ]
    ]
    metablocking: Optional[Union[Metablocking]] = Field(
        None, description="Metablocking method used in the workflow"
    )
    matching: Matching = Field(..., description="Matching method used in the workflow")
    clustering: Optional[Union[Clustering]] = Field(
        None, description="Clustering method used in the workflow"
    )


class FAISSWorkflow(BaseModel):
    """Schema for FAISS workflow"""

    type: Literal["faiss"]

    blocking: SchemaFAISSBlocking = Field(
        ..., description="Blocking method used in the workflow"
    )
    matching: Matching = Field(..., description="Matching method used in the workflow")
    clustering: Optional[Clustering] = Field(
        None, description="Clustering method used in the workflow"
    )


class Flat(BaseModel):
    """Schema for FAISS Flat blocking method"""

    type: Literal["flat"]
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of nearest neighbors to retrieve for each query vector",
    )


class IVF(Flat):
    """Schema for FAISS IVF blocking method"""

    type: Literal["ivf"]

    nlist: int = Field(
        32, ge=1, le=1000, description="Number of clusters to create in the IVF index"
    )
    nprobe: int = Field(
        8,
        ge=1,
        le=1000,
        description="Number of clusters to probe during search in the IVF index",
    )


class HNSW(Flat):
    """Schema for FAISS HNSW blocking method"""

    type: Literal["hnsw"]

    hnsw_m: int = Field(
        32, ge=1, le=200, description="Number of neighbors to connect in the HNSW graph"
    )


class SchemaFAISSBlocking(BaseModel):
    """Schema for FAISS blocking method"""

    method: Annotated[Union[IVF, Flat, HNSW], Field(discriminator="type")] = Field(
        ..., description="Blocking method used in the workflow"
    )


class BruteForceWorkflow(BaseModel):
    """Schema for Brute Force workflow"""

    type: Literal["brute_force"]

    matching: Matching = Field(..., description="Matching method used in the workflow")
    clustering: Optional[Clustering] = Field(
        None, description="Clustering method used in the workflow"
    )


class GroundTruth(BaseModel):
    """Schema for ground truth file"""

    file: UploadFile = File(..., description="Ground truth file for the workflow")
    separator: str = Field(..., description="Separator used in the ground truth file")


class PrivJedAIWorkflow(BaseModel):
    """Schema for privJedAI workflow"""

    workflow_id: str = Field(..., description="Unique identifier for the workflow")
    workflow_name: str = Field(..., description="Name of the workflow")
    created_at: str = Field(..., description="Timestamp when the workflow was created")
    updated_at: str = Field(
        ..., description="Timestamp when the workflow was last updated"
    )
    user_id: str = Field(
        ..., description="Identifier of the user who created the workflow"
    )

    workflow_config: Annotated[
        Union[LSHWorkflow, FAISSWorkflow, BruteForceWorkflow],
        Field(discriminator="type"),
    ]
