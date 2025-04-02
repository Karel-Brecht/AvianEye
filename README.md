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

### Installation on mac or Ubuntu
There seems to be an easier way for ubuntu and macOS by using respectively `apt install ffmpeg` or `brew install ffmpeg`.
But i haven't tried this out myself.


# YouTube Video Input
Implemented downloading from youtube in `app\operations\download.py`

Using the yt-dlp package. This is, up to today, a very well maintained package, and has lots of contributors and recent contributions.

# Bird detection
Implemented bird detection in `app\operations\detection.py`

Using `yolov10n`, a lightweight and fast pre-trained object-detection model.

model weights downloaded at `models/detection_model`

# Run the program

In your terminal change directory the cloned repository and activate your python environment (see above). 

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

This is a fine-tuned version of the google/efficientnet-b2 model. Aclassifier trained on an augmented version of the _Birds 525_ dataset.
The link to the original dataset could not be found but is most likely the same as the one from the following link: [https://huggingface.co/datasets/chriamue/bird-species-dataset/tree/main/data](https://huggingface.co/datasets/chriamue/bird-species-dataset/tree/main/data). This dataset contains 525 different species, 84.6k training samples, 2.63k validation samples and 2.63 test samples.

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

- Allow arbitrary aspect ratios, for now always resized to 1280x750
- Youtube download should probably not be full-res, already download downscaled, appropriate version
- Improve object tracking to not create multiple tracks for the same object.
    - Some possible approaches are listed as TODOs in the tracking file but especially this one: _allow multiple merges with the same new_track_id (tracks_to_merge.values()) but in a second pass allow only the ones with the shortest gap_size_
- Parametrize the remaining hard-coded parameters, marked with TODOs
- Group all processing parameters togetter in a clear config file
- Play around with parameters and optimize for the scope of target videos
- Fine tune the detection and classification models
    - Possibly create a dataset using the last best version of this program so that you can leverage the interpolated observations to get even better fine-tuned models.
- Put extra measures for deleting duplicate observations
    - Removing observations of which the corners allign too well
    - Removing observations that, over time, are too similar. (right now only duplicates are removed by if the start of a track is too similar to an already existing track)
- Remove tracks that are too ambiguous over time: e.g. the class probabilities differ too much over time
- Extract frames faster from video
- Dynamically load video frames for larger files
    - Right now the program will most likely crash when video files are too large.
- Enable parallellization for detections and classifications
- Provide option to do the detections on a lower frame rate and interpolate them.
- Write python tests!

# Deployment Strategy

## If I were to deploy this in a production environment.
- Containerize the service in a container with the correct environment installed.
- Load balancing: -> Kubernetes? Nginx?
- Downscale to an appropriate video resolution
- Handle arbitrary resolutoins and aspect ratios
- Make logs of video's processed, video duration, nr frames, processing time, ...
- Look into licenses of the used models, if they can be used for the production environment
- Optimize model parameters to match desired false-positive - false-negative ratio
- Process summary statistics to give a more realistic output (e.g. limit possible species to a curated list of species)
- Process video in chunks
- Currently this program relies a lot on the yt-dlp package, there have been occurences that these kind of packages stop working due to Changes on YouTube's side. It is important to use a well maintained package.
- Provide a simple API
- Call it BirdsAI or BirdsAIview

## Adaptations for Real-time analysis
- Video download and frame extraction should be handled differently.
- Downscale to a suitable resolution and frame-rate
- Do detections on an even lower frame-rate and interpolate the observations
- Implement causal tracking (without knowing the next frames)
- AI model inference must be fast, use faster models
    - e.g. use a fine-tuned end-to-end yolo model for doing detections and classifications simmulateously
    - Train lighter and faster models on accurately labeled data provided with the stronger and slower models
    - When using seperate detection and classification, implement classification of the detections in parallel
- Maybe rewrite (part of) the code in C++
