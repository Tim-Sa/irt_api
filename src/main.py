from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException

from .irt.irt import irt
from .irt.utils import df_consist_only_of, open_xlsx

from .config import avaliable_mime_types

app = FastAPI()


@app.post("/upload")
async def get_logits_by_file(file: UploadFile):

    # check the content type (MIME type)
    if file.content_type not in avaliable_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # IRT exec
    file = await file.read()
    df = open_xlsx(file)

    if not df_consist_only_of(df, set([0, 1])):
        raise HTTPException(status_code=400, detail="Table must contain only one and zero values")
    # TODO: check pipeline for test results table.

    try:
        irt_result = irt(df)
        return vars(irt_result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Can't process this data:\n\t{e}")
    
