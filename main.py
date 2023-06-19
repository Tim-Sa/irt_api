from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
        # check the content type (MIME type)
    if file.content_type not in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # do something with the valid file
    return {"filename": file.filename}