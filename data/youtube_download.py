import os
import yt_dlp
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display


def parse_lab_file(lab_file_path):
    """
    Parse a .lab file to extract the title and artist.

    Args:
        lab_file_path (str): The path to the .lab file to parse.

    Returns:
        tuple: The title and artist extracted from the file, or (None, None) if not found.
    """
    print(f"Attempting to parse .lab file: {lab_file_path}")
    title = artist = None
    try:
        with open(lab_file_path, 'r') as file:
            for line in file:
                if line.startswith("# title:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("# artist:"):
                    artist = line.split(":", 1)[1].strip()
                if title and artist:
                    break
        if not title or not artist:
            print(f"Missing title or artist in {lab_file_path}")
    except FileNotFoundError:
        print(f"File not found: {lab_file_path}")
    except Exception as e:
        print(f"Error reading .lab file {lab_file_path}: {e}")
    return title, artist


def download_audio_from_youtube(query, output_path):
    """
    Search for a song on YouTube and download its audio as an MP3 file.

    Args:
        query (str): The search query for YouTube (e.g., song title and artist).
        output_path (str): The path to save the downloaded MP3 file.

    Returns:
        str: The path to the downloaded file, or None if the download fails.
    """
    print(f"Initiating YouTube download for query: {query}")

    progress_bar = IntProgress(value=0, min=0, max=100, description="Downloading:", bar_style="info")
    progress_label = HTML(value="Preparing download...")
    progress_box = VBox([progress_label, progress_bar])
    display(progress_box)

    def download_hook(d):
        """
        Hook function for yt_dlp to integrate with ipywidgets progress bar.
        """
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                progress_bar.max = total
                progress_bar.value = downloaded
                progress_label.value = f"{(downloaded / total) * 100:.2f}% downloaded"
        elif d['status'] == 'finished':
            progress_bar.bar_style = "success"
            progress_label.value = "Download complete!"
        elif d['status'] == 'error':
            progress_bar.bar_style = "danger"
            progress_label.value = "Error during download!"

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': output_path,
        'progress_hooks': [download_hook],  # Use custom hook for progress
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
            if 'entries' in search_results and search_results['entries']:
                video_url = search_results['entries'][0]['url']
                print(f"Found video URL: {video_url}. Starting download...")
                ydl.download([video_url])
                return output_path
            else:
                progress_label.value = "No results found on YouTube for the query."
    except Exception as e:
        progress_label.value = f"Error: {e}"
    return None


def process_lab_files(base_directory):
    """
    Traverse through folders in the base directory, process .lab files, and download audio.

    Args:
        base_directory (str): The root directory to start processing .lab files.
    """
    print(f"Starting to process .lab files in base directory: {base_directory}")
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".txt"):
                print(f"\n\n\nEntering directory: {root}")
                lab_file_path = os.path.join(root, file)
                print(f"Processing .lab file: {lab_file_path}")

                file_root, file_extension = os.path.splitext(file)

                # Parse the .lab file
                title, artist = parse_lab_file(lab_file_path)
                if not title or not artist:
                    print(f"Skipping {lab_file_path} due to missing metadata.")
                    continue

                query = f"{title} {artist}"
                output_mp3_path = os.path.join(root, f"{file_root} ({artist} - {title}).mp3")
                print(f"Expected output path for MP3: {output_mp3_path}")

                # Check if the MP3 already exists
                if os.path.exists(output_mp3_path):
                    print(f"Audio file already exists: {output_mp3_path}. Skipping download.")
                    continue

                # Download audio
                download_audio_from_youtube(query, output_mp3_path)
