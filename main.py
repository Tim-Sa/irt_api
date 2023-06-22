import json

import pandas as pd

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException

from settings import avaliable_mime_types
from irt.irt import irt

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # check the content type (MIME type)
    if file.content_type not in avaliable_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # IRT exec
    file = await file.read()
    df = pd.read_excel(file, index_col=0)
    irt_result = irt(df)

    return json.dumps(vars(irt_result))