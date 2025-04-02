# Installation

Clone this repository and install following dependencies:

## Python dependencies

Installed packages and dependencies can be found in _requirements.txt_

I have used a `python-venv` in windows with `Python 3.12.3`

### Python

Have python 3.9 or higher installed. Mine was `Python 3.12.3`

### Pip

Update pip:

```bash
python -m pip install --upgrade pip
```

### yt-dlp

Install yt-dlp:
```bash
pip install yt-dlp
```
I have version _2025.3.27_ installed.

### opencv

Install opencv:
```bash
pip install opencv-python
```
I have vesrion _4.11.0.86_ installed.

### Ultralytics

Install Ultralytics:
```bash
pip install ultralytics
```
I have version _8.3.98_ installed.

### transformers

Install transformers
```bash
pip install transformers
```

I have version _4.50.3_ installed.

## ffmpeg
The project relies on ffmpeg to merge audio and video files together.

### Installation on windows.
Install from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

You can follow this guide: [https://phoenixnap.com/kb/ffmpeg-windows](https://phoenixnap.com/kb/ffmpeg-windows)
- Download the zip file from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
    - I have dowloaded the _ffmpeg-git-full.7z_ from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
- Extract the zip file
- Place folder in your root directory: `C:\\` 
- Rename the folder to ffmpeg
- Add FFmpeg to environment Variable Path under User variables : add `c:\\ffmpeg\bin`
- Restart your vscode or terminal for the changes to take effect.
- Verify correct environment by executing the following command in the terminal `ffmpeg`

### On mac or Ubuntu
There seems to be an easier way for ubuntu and macOS by using respectively `apt install ffmpeg` or `brew install ffmpeg`.
But i haven't tried this out myself.


# YouTube Video Input
Implemented downloading from youtube in `app\operations\download.py`

Using the yt-dlp package. This is, up to today, a very well maintained package, and has lots of contributors and recent contributions.

# Bird detection
Implemented bird detection in `app\operations\detection.py`

Using yolov10n

A lightweight and fast pre-trained object-detection model.

model weights downloaded at `models/detection_model`

# Run

Run the program with:
```bash
python ./app/main.py yt-link -o output_file_path.mp4
```

Both arguments are optional, defaulting to the following
```bash
python ./app/main.py https://www.youtube.com/watch?v=swavE6ZwLJQ -o ./output.mp4
```

The output mp4 file is for the moment scaled to (1280, 720) pixels and contains the original audio.
The detections and species counts are annotated on the video.

A json file will be stored to `./summary_statistics.json`. This contains how many times each bird species has appeared throughout the whole video, along with the average confidence for each species.


# Species classification
Implemented species classification in `app\operations\classification.py`.

Using the following model from huggingface: [https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2](https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2)

Fine tuned version of the google/efficientnet-b2 model. Aclassifier trained on an augmented version of the _Birds 525_ dataset.
The link to the original dataset could not be found but is most likely the same as the one on the following link: [https://huggingface.co/datasets/chriamue/bird-species-dataset/tree/main/data](https://huggingface.co/datasets/chriamue/bird-species-dataset/tree/main/data). This dataset contains 525 different species, 84.6k training samples, 2.63k validation samples and 2.63 test samples.

The model is said to have the following accuracy on the dataset:
- Training: 0.999480
- Validation: 0.985904
- Test: 0.991238

More info about the model can be found on the webpage, of which I have included a copy at `models\classification_model\model_webpage`

model downloaded at `models/classification_model`

# Object tracking
Implemented object tracking in `app\operations\tracking.py` with lots of help from _claude 3.7 Sonnet_.

# Next Steps

Potential improvements and future work to enhance capabilities and performance.

# Deployment Strategy

If I were to deploy this in a production environment.
- Containerize the service in a container with the correct environment installed.
- Load balancing: -> Kubernetes? Nginx?
- Choose appropriate video resolution to process
- Handle different resolutoins and aspect ratios
- Make logs of video's processed, video duration, nr frames, processing time
- Scalable?
- Look into licenses of the chosen model, if it can be used for the production environment
- Optimize model parameters, e.g. set confidence to match desired false-positive - false-negative ratio
- Process video in chuncks?
- Relies a lot on the yt-dlp package, there have been occurences that these kind of packages stop working due to Changes on YouTube's side. Important to use a well maintained package.
- Provide simple API
- Call it BirdsAI or BirdsAIview

Adaptations for Real-time analysis
- Process video in chuncks
- Inference must be fast, not every model can be used
- e.g. YOLOv10 should be faster than YOLOv8
- Maybe process on lower resolution and skip some frames
- Train lighter and faster model on accurately labeled data provided with the stronger and slower model


