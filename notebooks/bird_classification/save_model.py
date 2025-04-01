from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# Load the model and preprocessor
preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

# Save them locally
preprocessor.save_pretrained("./local_model")
model.save_pretrained("./local_model")