# Use a pipeline as a high-level helper
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(
        self,
        path : str = "Salesforce/blip-image-captioning-base"
    ):
        self.processor = BlipProcessor.from_pretrained(path)
        self.model = BlipForConditionalGeneration.from_pretrained(path)
    
    def caption(self, text = "", img = None):
        assert img != None, "Img cannot be None, check your request"
        inputs = self.processor(img, text, return_tensors="pt")
        out = self.model.generate(**inputs)

        return self.processor.decode(out[0], skip_special_tokens=True)

