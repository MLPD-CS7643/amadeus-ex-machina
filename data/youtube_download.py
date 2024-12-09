import os
import yt_dlp
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display
import mutagen
import shutil
import subprocess
import json
import re


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
    for root, dirs, _ in os.walk(base_directory):
        for dir in sorted(dirs):
            base_dir = os.path.join(root, dir)
            for _, _, files in os.walk(base_dir):
                for file in files:
                    if file == "salami_chords.txt":
                        print(f"\nEntering directory: {root}")

                        lab_file_path = os.path.join(f"{root}{dir}", file)
                        print(f"Processing .lab file: {lab_file_path}")

                        # Parse the .lab file
                        title, artist = parse_lab_file(lab_file_path)
                        if not title or not artist:
                            print(f"Skipping {lab_file_path} due to missing metadata.")
                            continue

                        query = f"{title} {artist}"
                        output_mp3_path = f"{base_dir}{os.path.sep}{dir} ({artist} - {title}).mp3"
                        print(f"Expected output path for MP3: {output_mp3_path}")

                        # Check if the MP3 already exists
                        if os.path.exists(output_mp3_path):
                            # Check for 0-byte file
                            if os.path.getsize(output_mp3_path) == 0:
                                print(f"File {output_mp3_path} is 0 bytes. Deleting it.")
                                os.remove(output_mp3_path)
                            else:
                                print(f"Audio file already exists: {output_mp3_path}. Skipping download.")
                                continue
                        else:
                            print(f"Audio file does not exist: {output_mp3_path}")

                        # Download audio
                        download_audio_from_youtube(query, output_mp3_path.replace(".mp3", ""))


def process_downloaded_songs(billboard_data_directory, output_path, threshold=1):
    """
    Process downloaded songs by checking their length against their lab file and only accepting songs whose length is within an acceptable threshold
    in comparison with their respective lab files.

    Args:
        billboard_data_directory (str): The directory containing the downloaded songs.
        output_path (str): The path to the output file.
        threshold (int): The acceptable threshold in seconds for the duration mismatch between the MP3 and lab file.

    Returns:
        None
    """
    print("Processing downloaded songs...")

    accepted_dir = f"{output_path}accepted"
    rejected_dir = f"{output_path}rejected"
    missing_mp3 = f"{rejected_dir}{os.path.sep}missing_mp3"
    zero_bytes_dir = f"{rejected_dir}{os.path.sep}zero_bytes_dir"
    missing_salami_chords_dir = f"{rejected_dir}{os.path.sep}missing_salami_chords"
    missing_salami_chords_duration_dir = f"{rejected_dir}{os.path.sep}missing_salami_chords_duration"
    duration_mismatch_dir = f"{rejected_dir}{os.path.sep}duration_mismatch"
    processing_result_file = f"{output_path}processing_result.json"

    # Create output directory if it does not exist
    for dir in [accepted_dir, rejected_dir, missing_mp3, zero_bytes_dir, missing_salami_chords_dir, missing_salami_chords_duration_dir, duration_mismatch_dir]:
        os.makedirs(dir, exist_ok=True)
    res = {"accepted": 0, "rejected": {"missing_mp3": 0, "zero_bytes": 0, "missing_salami_chords": 0, "missing_salami_chords_duration": 0, "duration_mismatch": 0}}

    for dir in [accepted_dir, rejected_dir]:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)  # Delete file
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    shutil.rmtree(dir_path)  # Delete subdirectory
        else:
            print(f"The folder '{dir}' does not exist, no need to delete its contents.")

    # Delete processing result file if it exists
    if os.path.exists(processing_result_file):
        os.remove(processing_result_file)

    for _, outer_dirs, _ in os.walk(billboard_data_directory):
        for dir in sorted(outer_dirs):
            # print(f"\nEntering directory: {dir}")
            for _, _, files in os.walk(f"{billboard_data_directory}{os.path.sep}{dir}"):
                for file in files:
                    if file.endswith(".mp3"):
                        mp3_file_path = os.path.join(f"{billboard_data_directory}{dir}", file)
                        # print(f"\nProcessing MP3 file: {mp3_file_path}")
                        lab_file_path = os.path.join(f"{billboard_data_directory}{dir}", "salami_chords.txt")
                        # print(f"Processing lab file: {lab_file_path}")
                        if not os.path.exists(lab_file_path):
                            print(f"Skipping {mp3_file_path} due to missing salami_chords file.")
                            # copy directory to rejected folder
                            shutil.copytree(f"{billboard_data_directory}{dir}", f"{missing_salami_chords_dir}{os.path.sep}{dir}")
                            # create txt file in new folder directory containing reason for rejection
                            with open(f"{missing_salami_chords_dir}{os.path.sep}{dir}{os.path.sep}rejection_reason.txt", "w") as f:
                                f.write("Missing salami chords file")
                            res["rejected"]["missing_salami_chords"] += 1
                            continue

                        if not os.path.exists(mp3_file_path):
                            print(f"Skipping {lab_file_path} due to missing MP3 file.")
                            # copy directory to rejected folder
                            shutil.copytree(f"{billboard_data_directory}{dir}", f"{missing_mp3}{os.path.sep}{dir}")
                            # create txt file in new folder directory containing reason for rejection
                            with open(f"{missing_mp3}{os.path.sep}{dir}{os.path.sep}rejection_reason.txt", "w") as f:
                                f.write("Missing mp3 file")
                            res["rejected"]["missing_mp3"] += 1
                            continue

                        # Get the duration of the MP3 file
                        mp3_duration = get_mp3_duration(mp3_file_path)
                        if mp3_duration is None:
                            print(f"Skipping {mp3_file_path} due to missing mp3 duration.")
                            # copy directory to rejected folder
                            shutil.copytree(f"{billboard_data_directory}{dir}", f"{zero_bytes_dir}{os.path.sep}{dir}")
                            with open(f"{zero_bytes_dir}{os.path.sep}{dir}{os.path.sep}rejection_reason.txt", "w") as f:
                                f.write("Missing duration in mp3 file (probably 0 byte file)")
                            res["rejected"]["zero_bytes"] += 1
                            continue

                        # Get the duration from the .lab file
                        lab_duration = get_lab_duration(lab_file_path)
                        if lab_duration is None:
                            print(f"Skipping {mp3_file_path} due to missing .txt duration.")
                            # copy directory to rejected folder
                            shutil.copytree(f"{billboard_data_directory}{dir}", f"{missing_salami_chords_duration_dir}{os.path.sep}{dir}")
                            with open(f"{missing_salami_chords_duration_dir}{os.path.sep}{dir}{os.path.sep}rejection_reason.txt", "w") as f:
                                f.write("Missing duration in salami_chords file")
                            res["rejected"]["missing_salami_chords_duration"] += 1
                            continue

                        # Check if the durations are within an acceptable threshold
                        if abs(mp3_duration - lab_duration) > threshold:
                            print(f"Skipping {mp3_file_path} due to duration mismatch: {round(abs(mp3_duration-lab_duration), 3)}s difference")
                            # copy directory to rejected folder
                            shutil.copytree(f"{billboard_data_directory}{dir}", f"{duration_mismatch_dir}{os.path.sep}{dir}")
                            with open(f"{duration_mismatch_dir}{os.path.sep}{dir}{os.path.sep}rejection_reason.txt", "w") as f:
                                f.write("Duration mismatch")
                            res["rejected"]["duration_mismatch"] += 1
                            continue

                        print(f"Accepted {mp3_file_path}: {round(abs(mp3_duration-lab_duration), 3)}s difference")
                        shutil.copytree(f"{billboard_data_directory}{dir}", f"{accepted_dir}{os.path.sep}{dir}")
                        res["accepted"] += 1
    # Create a json file in the processed directory containing the results
    with open(processing_result_file, "w") as f:
        json.dump(res, f, indent=4)

def get_mp3_duration(mp3_file_path):
    """
    Get the duration of an MP3 file using the mutagen library, fixing encoding issues if needed.

    Args:
        mp3_file_path (str): The path to the MP3 file.

    Returns:
        float: The duration of the MP3 file in seconds, or None if the duration cannot be determined.
    """
    try:
        from mutagen.mp3 import MP3
        
        # Check if the file exists
        if not os.path.exists(mp3_file_path):
            # print(f"Error: File {mp3_file_path} does not exist.")
            return None

        try:
            # Try reading the MP3 file with mutagen
            audio = MP3(mp3_file_path)
            return audio.info.length
        except Exception as e:
            # print(f"Error reading MP3 file {mp3_file_path}: {e}")
            # print("Attempting to fix the encoding using ffmpeg...")

            # Fix encoding using ffmpeg
            fixed_file_path = f"{mp3_file_path}.fixed.mp3"
            command = [
                "ffmpeg", "-y", "-i", mp3_file_path, 
                "-acodec", "copy", fixed_file_path
            ]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Replace the original file with the fixed file
            if os.path.exists(fixed_file_path):
                os.replace(fixed_file_path, mp3_file_path)
                # print(f"File fixed and saved as {mp3_file_path}. Retrying duration calculation...")

                # Retry reading the fixed MP3 file
                audio = MP3(mp3_file_path)
                return audio.info.length
            else:
                # print("Failed to fix the MP3 file with ffmpeg.")
                return None

    except Exception as e:
        # print(f"Error getting duration for {mp3_file_path}: {e}")
        return None

def get_lab_duration(lab_file_path):
    """
    Get the time value on the last line of a .lab file next to 'end'.

    Args:
        lab_file_path (str): The path to the .lab file.

    Returns:
        float: The time value in seconds, or None if the time cannot be determined.
    """
    try:
        with open(lab_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                # Read lines in reverse
                for line in reversed(lines):
                    if line.strip():
                        # Use regex to retrieve all numerical data on this line
                        match = re.search(r'\d+(?:\.\d+)?', line)
                        if match:
                            # Convert the time to seconds
                            time_str = match.group()
                            return float(time_str)
        return None
    except Exception as e:
        # print(f"Error getting last time from .lab file {lab_file_path}: {e}")
        return None