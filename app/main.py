import os
import logging
from datetime import datetime

import pandas as pd
from irt_test.irt import irt

from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

avaliable_mime_types = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='irt_api_errors.log', 
    level=logging.ERROR
)

origins = [
    f"http://{os.getenv('FRONTEND_HOST')}",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/irt")
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
    