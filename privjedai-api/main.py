"""REST API for privjedAI"""

import traceback
import os
import io
import uuid
from typing import Optional
import pandas as pd
from fastapi.params import Depends
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from schemas import SchemaBloomFilterConfig, PrivJedAIWorkflow
from utils import (
    bloom_filter_config_form,
    load_encoded_data,
    execute_workflow_utils,
    # parse_lsh_config,
    parse_privjedai_workflow,
)
from privjedai.encoder import BloomFilterConfig, BloomFilter

app = FastAPI(title="PrivjedAI API", description="API for PrivjedAI", version="1.0.0")

# Ensure the storage directory exists when the server starts
STORAGE_DIR = "encoded_data"
CONFIG_DIR = "config_data"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


@app.post("/encode")
async def encode_dataset(
    background_tasks: BackgroundTasks,
    bloom_filter_config: SchemaBloomFilterConfig = Depends(bloom_filter_config_form),
):
    """This function is used to encode a dataset using the specified Bloom filter configuration."""
    # 1. Read the uploaded file and setup config
    file_bytes = await bloom_filter_config.file.read()
    pandas_df = pd.read_csv(io.BytesIO(file_bytes), sep=bloom_filter_config.separator)

    config = BloomFilterConfig(
        size=bloom_filter_config.bloom_filter_size,
        num_hashes=bloom_filter_config.number_of_hashes,
        offset=0,
        hashing_type=bloom_filter_config.hashing_type,
        qgrams=bloom_filter_config.qgrams,
        salt=bloom_filter_config.salt,
        attributes=bloom_filter_config.attributes,
    )

    # 2. Generate the file synchronously
    temp_pkl = f"encoded_bloom_{uuid.uuid4().hex}.pkl"
    bloom_generator = BloomFilter(config)
    encoded_d1 = bloom_generator.encode(pandas_df)
    encoded_d1.to_file(temp_pkl)

    # 3. Tell FastAPI to delete the file AFTER the response is sent
    background_tasks.add_task(os.remove, temp_pkl)

    # 4. Return the file to the user
    return FileResponse(
        path=temp_pkl,
        filename="bloom_filter.pkl",
        media_type="application/octet-stream",
    )


@app.post("/workflow/execute")
async def execute_privjedai_workflow(
    encoded_d1: UploadFile = File(..., description="First encoded data file"),
    encoded_d2: UploadFile = File(..., description="Second encoded data file"),
    ground_truth: Optional[UploadFile] = File(
        None, description="Optional ground truth file"
    ),
    separator_ground_truth: Optional[str] = Form(
        ",", description="Separator for the ground truth file"
    ),
    config_json: PrivJedAIWorkflow = Depends(parse_privjedai_workflow),
):

    """
    Execute the PrivjedAI workflow with the provided encoded data and configuration.
    """

    temp_p1_path = None
    temp_p2_path = None
    try:
        bloom_encoded_data, temp_p1_path, temp_p2_path = await load_encoded_data(
            encoded_d1, encoded_d2
        )
        if ground_truth:
            ground_truth_index = await ground_truth.read()
            ground_truth_index_df = pd.read_csv(
                io.BytesIO(ground_truth_index), sep=separator_ground_truth
            )
            bloom_encoded_data.set_ground_truth(ground_truth_index_df)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read data files: {str(e)}"
        ) from e
    finally:
        if temp_p1_path and os.path.exists(temp_p1_path):
            os.remove(temp_p1_path)
        if temp_p2_path and os.path.exists(temp_p2_path):
            os.remove(temp_p2_path)

    try:
        execution, pairs_df = execute_workflow_utils(
            config=config_json, encoded_data=bloom_encoded_data
        )

    except Exception as e:
        # This prints the full stack trace to your terminal
        print("Error executing workflow!")
        traceback.print_exc()

        # Optional: Return a 500 error instead of letting it return 200 OK
        raise HTTPException(
            status_code=500, detail="Workflow execution failed. Check server logs."
        ) from e
    csv_string = pairs_df.to_csv(index=False)

    return JSONResponse(content={"execution": execution, "pairs": csv_string})
