import os
import yt_dlp

DOWNLOAD_PATH = 'downloads/videos'
FILENAME = 'download.mp4'
PATH_DOWNLOAD_FILE = f'{DOWNLOAD_PATH}/{FILENAME}'
YDL_OPTIONS = {
    'outtmpl': PATH_DOWNLOAD_FILE,
}

def download_video(video_url: str) -> str:
    """
    Downloads a YouTube video at the highest available resolution and saves it as an MP4 file.

    Args:
        video_url (str): The URL of the YouTube video.
    
    Returns:
        str: The file path of the downloaded video.
    
    Raises:
        ValueError: If the video cannot be downloaded.
    """
    # First delete the file if it exists
    delete_video()

    # Now download the video
    with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
        try:
            info_dict = ydl.extract_info(video_url, download=True)
            file_path = ydl.prepare_filename(info_dict)
            return file_path
        except Exception as e:
            raise ValueError(f"Error downloading video: {e}")

def delete_video():
    """
    Deletes the downloaded video file if it exists.

    Returns:
        bool: True if the file was deleted, False if it did not exist.
    Raises:
        ValueError: If there was an error deleting the file.
    """
    if os.path.exists(PATH_DOWNLOAD_FILE):
        try:
            os.remove(PATH_DOWNLOAD_FILE)
            return True
        except Exception as e:
            raise ValueError(f"Error deleting existing file: {e}")
    else:
        return False
