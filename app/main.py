import logging
from datetime import datetime
from typing import Dict

import pandas as pd
from irt_test.irt import irt

from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


class IrtModel(BaseModel):
    '''
    subj1: {
        "task1": 0
        "task2": 1
    }

    subj2: {
        "task1": 0
        "task2": 1   
    }
    '''
    subjects: Dict[str, Dict[str, int]]


avaliable_mime_types = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='irt_api_errors.log', 
    level=logging.ERROR
)


origins = ["*"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def get_logits_by_file(file: UploadFile):

    if file.content_type not in avaliable_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type (available file type is '.xlsx')")
    
    file = file.file.read()

    try:

        df = pd.read_excel(file, index_col=0)

    except Exception as e:

        detail = "Can't open this file."
        err_msg = f"\n{datetime.now()}\n{detail}\nError:\n\t{e}\n"

        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=detail)

    try:

        irt_result = irt(df)
        irt_result_dict = vars(irt_result)

        return irt_result_dict
    
    except Exception as e:
        
        detail ="Can't process this data."
        err_msg = f"\n{datetime.now()}\n{detail}\nError:\n\t{e}\n"

        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=detail)
    

@app.post("/irt")
async def get_logits_by_json(irt_info: IrtModel):

    try:    

        df = pd.DataFrame(irt_info.dict()["subjects"])   

    except Exception as e:

        detail = "Can't open this file."
        err_msg = f"\n{datetime.now()}\n{detail}\nError:\n\t{e}\n"

        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=detail)

    try:
        df = df.T
        irt_result = irt(df)
        irt_result_dict = vars(irt_result)

        return irt_result_dict
    
    except Exception as e:
        
        detail ="Can't process this data."
        err_msg = f"\n{datetime.now()}\n{detail}\nError:\n\t{e}\n"

        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=detail)
    