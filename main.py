import pandas as pd
from irt_test.irt import irt

from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException

avaliable_mime_types = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

app = FastAPI()


@app.post("/irt")
async def get_logits_by_file(file: UploadFile):

    # check the content type (MIME type)
    if file.content_type not in avaliable_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        file = file.file.read()
        df = pd.read_excel(file, index_col=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Can't open this file:\n\t{e}")

    try:
        irt_result = irt(df)
        irt_result_dict = vars(irt_result)
        return irt_result_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Can't process this data:\n\t{e}")
    