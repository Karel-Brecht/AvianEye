# Installation

Installed packages and dependencies can be found in _requirements.txt_

I have used a python-venv in windows

Install yt-dlp:
```bash
pip install yt-dlp
```
I have version _2025.3.27_ installed.

Install opencv:
```bash
pip install opencv-python
```
I have vesrion _4.11.0.86_ installed.

Install Ultralytics:
```bash
pip install ultralytics
```
I have version _8.3.98_ installed.


# YouTube Video Input
For youtube video input I am using the yt-dlp package. This is, up to today, very well maintained, and has lots of contributors and recent contributions.

Install yt-dlp:

```bash
pip install yt-dlp
```

I have version _2025.3.27_ installed.

# Endt-to-End object detection & classification
Using yolo

Install yolo:

```bash
pip install ultralytics
```

I have version _8.3.98_ installed.

# ffmpeg

## Installation on windows.
[text](https://ffmpeg.org/download.html)
following this guide:
[text](https://phoenixnap.com/kb/ffmpeg-windows)
- Download zip file
- Extract zip file
- Rename to ffmpeg
- Add FFmpeg to environment Variable Path under User variables : add `c:\\ffmpeg\bin`
- Restart your vscode or terminal for the changes to take effect.
- Verify correct environment by executing the following command in the terminal 
```bash
ffmpeg
```

## On mac or Ubuntu
There seems to be an easier way for ubuntu and macOS by using respectively `apt install ffmpeg` or `brew install ffmpeg`.
But i haven't tried this out myself.

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