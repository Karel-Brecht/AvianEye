import argparse
from downloader import download_video


DEFAULT_YT_URL = "https://www.youtube.com/watch?v=swavE6ZwLJQ"

def main():

    # Optional youtube video URL
    parser = argparse.ArgumentParser(description="Download and process a YouTube video.")
    parser.add_argument('yt_url', type=str, nargs='?', default=DEFAULT_YT_URL, help = "Optional YouTube video URL")
    args = parser.parse_args()
    yt_url = args.yt_url
    print(f"Downloading video from: {yt_url}")

    download_path = download_video(yt_url)
    print(f"Video downloaded to: {download_path}")

if __name__ == "__main__":
    main()
