# Importing the libraries needed
import torch
import urllib.request
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# Determining the file URL
# url = 'https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTxuWacbqJQhZOe3GKrV7mGiLRFkxb4MuPQpDrZfvEhtx3Iilkw-gfxU-r_O5RH4u2vui6UtOJu25B-dOrUTlEbYQ'
# url = 'https://tx.audubon.org/sites/default/files/styles/bean_wysiwyg_full_width/public/cbcpressroom_tuftedtitmouse-judyhowle.jpg?itok=VMtDnqip'

# # Opening the image using PIL
# img = Image.open(urllib.request.urlretrieve(url)[0])

# try open image from ./bird_imgs/image.png
try:
    img = Image.open("bird_imgs/bluejay.png")
except FileNotFoundError:
    print("Image file not found. Please check the path.")
    exit()
# convert opened image to RGB
img = img.convert("RGB")


# # Loading the model and preprocessor from HuggingFace
# preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
# model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

# Loading the model and preprocessor from local directory
preprocessor = EfficientNetImageProcessor.from_pretrained("models/classification_model")
model = EfficientNetForImageClassification.from_pretrained("models/classification_model")

# Preprocessing the input
inputs = preprocessor(img, return_tensors="pt")

# Running the inference
with torch.no_grad():
    logits = model(**inputs).logits

# Getting the predicted label
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
