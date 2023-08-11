from fastapi import FastAPI
from explainer.model import ImageCaptioner
from pydantic import BaseModel
from PIL import Image
import requests


app = FastAPI()
captioner = ImageCaptioner()

class Item(BaseModel):
    url: str
    text: str


@app.get("/ping")
def ping():
    return {"ping": "I'm up!"}

@app.post("/predict")
def generate_caption(data : Item):
    raw_image = Image.open(requests.get(data.url, stream=True).raw).convert('RGB')
    return captioner.caption(text=data.text, img=raw_image)