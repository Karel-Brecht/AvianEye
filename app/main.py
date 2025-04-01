import argparse
import sys
from pathlib import Path

# Add parent directory to Python path to resolve imports
sys.path.append(str(Path(__file__).parent.parent))

from app.operations.download import Downloader
from app.operations.detection import BirdDetector
from app.operations.classification import BirdClassifier
from app.models.video_processor import VideoProcessor


DEFAULT_YT_URL = "https://www.youtube.com/watch?v=swavE6ZwLJQ"

DEFAULT_OUTPUT_PATH = "./outputvidrun.mp4"

def main():

    # Optional youtube video URL
    parser = argparse.ArgumentParser(description="Download and process a YouTube video.")
    parser.add_argument('yt_url', type=str, nargs='?', default=DEFAULT_YT_URL, help = "Optional YouTube video URL")
    parser.add_argument('-o', '--output', type=str, nargs='?', default=DEFAULT_OUTPUT_PATH, help="Optional output file path")
    args = parser.parse_args()
    yt_url = args.yt_url
    output_path = args.output
    print(f"Downloading video from: {yt_url}")
    print(f"Output will be saved to: {output_path}")

    downloader = Downloader()
    detector = BirdDetector("models/detection_model/yolov10n.pt")
    classfier = BirdClassifier("models/classification_model")

    vp = VideoProcessor(yt_url, downloader, detector, classfier)

    vp.run(output_path, frame_ids=True)

if __name__ == "__main__":
    main()
