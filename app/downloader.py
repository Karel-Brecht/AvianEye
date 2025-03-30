import yt_dlp

DOWNLOAD_PATH = 'downloads/videos'
FILENAME = 'download.mp4'
YDL_OPTIONS = {
    'outtmpl': f'{DOWNLOAD_PATH}/{FILENAME}',
}

def download_video(video_url: str) -> str:
    """
    Downloads a YouTube video at the highest available resolution and saves it as an MP4 file.

    Args:
        url (str): The URL of the YouTube video.
    
    Returns:
        str: The file path of the downloaded video.
    
    Raises:
        ValueError: If the video cannot be downloaded.
    """
    with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
        try:
            info_dict = ydl.extract_info(video_url, download=True)
            file_name = ydl.prepare_filename(info_dict)
            return file_name
        except Exception as e:
            raise ValueError(f"Error downloading video: {e}")
