from fastapi import Depends, FastAPI
from pydantic import BaseModel
from src.punctuazier import Punctuazier
from src.paragraphizer import Paragraphizer
from typing import List


app = FastAPI()


class RecoverRequest(BaseModel):
    text: str
    lang: str = None
    text_block_size: int = 100


class RecoverResponse(BaseModel):
    paragraphs: List[str]


@app.on_event("startup")
def startup_event():
    global punct_model, par_model
    punct_model = Punctuazier()
    par_model = Paragraphizer()


@app.get("/")
def index():
    return {"text": f"Text Structure Recovery: check {app.docs_url} for debug interface and functions"}


@app.post("/get_paragraphs")
def break_into_paragraphs(input: RecoverRequest):
    return RecoverResponse(
        paragraphs=par_model.breakdown(input.text)
    )


@app.post("/recover_structure")
def recover_structure_from_scratch(input: RecoverRequest):
    return RecoverResponse(
        paragraphs=par_model.breakdown(
            punct_model.punctuaze(input.text, lang=input.lang, len_limit=input.text_block_size))
    )


@app.post("/recover_punctuation")
def place_punctuation(input: RecoverRequest):
    return RecoverResponse(
        paragraphs=[
            punct_model.punctuaze(
                input.text, lang=input.lang, len_limit=input.text_block_size
            )
        ]
    )
