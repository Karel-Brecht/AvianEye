import os
import yt_dlp

class Downloader:
    def __init__(self, download_path='downloads/videos', filename='download.mp4'):
        self.download_path = download_path
        self.filename = filename
        self.file_path = os.path.join(download_path, filename)
        self.ydl_options = {
            'outtmpl': self.file_path,
        }

    def download_video(self, video_url: str) -> str:
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
        self.delete_video()

        # Now download the video
        with yt_dlp.YoutubeDL(self.ydl_options) as ydl:
            try:
                info_dict = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info_dict)
                return file_path
            except Exception as e:
                raise ValueError(f"Error downloading video: {e}")

    def delete_video(self) -> bool:
        """
        Deletes the downloaded video file if it exists.

        Returns:
            bool: True if the file was deleted, False if it did not exist.

        Raises:
            ValueError: If there was an error deleting the file.
        """
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                return True
            except Exception as e:
                raise ValueError(f"Error deleting existing file: {e}")
        else:
            return False
