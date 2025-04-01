import os
import tempfile
import yt_dlp

class Downloader:
    def __init__(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.download_path = self.temp_dir.name  # Store files here
        self.cleaned_up = False

        self.ydl_options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Ensures MP4 format
            'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),  # Saves as title.mp4
            'merge_output_format': 'mp4',  # Ensures final file is MP4 if merging is needed
        }

    def cleanup(self):
        """Remove temporary files and directory."""
        self.temp_dir.cleanup()
        self.cleaned_up = True # Mark as cleaned up

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

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
        with yt_dlp.YoutubeDL(self.ydl_options) as ydl:
            try:
                info_dict = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info_dict)
                return file_path
            except Exception as e:
                raise ValueError(f"Error downloading video: {e}")
