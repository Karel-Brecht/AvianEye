import torch
from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor
from PIL import Image

class BirdClassifier:
    def __init__(self, model_path: str):
        """Initialize the classifier with a pre-trained EfficientNet model and preprocessor."""
        try:
            # Attempt to load the preprocessor and model from the provided path
            self.preprocessor = EfficientNetImageProcessor.from_pretrained(model_path)
            self.model = EfficientNetForImageClassification.from_pretrained(model_path)
            self.class_names = self.model.config.id2label
            print("Model and preprocessor loaded successfully.")
        except Exception as e:
            print(f"Unexpected error while loading the model: {e}")
            raise
    
    def classify(self, image: Image.Image) -> tuple:
        """
        Classify a given image.
        
        Parameters:
            image (PIL.Image.Image): The input image to classify.
        
        Returns:
            predicted_class (str): The predicted class label.
            confidence (float): The confidence of the prediction.
            class_probabilities (dict): A dictionary of class names and their corresponding probabilities.
        """

        # Check if the input is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        
        # if RBGA convert to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Check if the image is in RGB format
        if image.mode != "RGB":
            raise ValueError("Image must be in RGB format.")
        
        # Preprocess the input image
        inputs = self.preprocessor(image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get the predicted label (class index)
        predicted_label = logits.argmax(-1).item()
        predicted_class = self.class_names[predicted_label]

        # Get all class_names with their corresponding probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
        class_probabilities = {self.class_names[i]: prob for i, prob in enumerate(probabilities)}
        
        # Return the predicted label and its associated confidence (optional)
        confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()  # Get the confidence of the prediction
        
        return predicted_class, confidence, class_probabilities
