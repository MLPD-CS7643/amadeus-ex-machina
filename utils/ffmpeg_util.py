import os
import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

FFMPEG_PATH = Path(__file__).parents[1] / 'bin' / 'ffmpeg' / 'ffmpeg.exe'

def convert_audio(input_file, output_folder, output_format):
    """Convert an audio file to a specified format using ffmpeg called via subprocess."""
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file_path = os.path.join(output_folder, f"{base_filename}.{output_format}")
    
    if output_format == 'mp3':
        command = [
            FFMPEG_PATH,
            '-i', input_file,
            #'-acoded', "libmp3lame",
            '-ab', '192k',
            output_file_path,
            '-y'  # Overwrite output files without asking
        ]
    else:
        command = [
            FFMPEG_PATH,
            '-i', input_file,
            output_file_path,
            '-y'  # Overwrite output files without asking
        ]
    
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def batch_convert(folder_path, input_ext, output_folder, output_format, n_jobs=1):
    """Convert all audio files in a folder from one format to another in parallel."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(input_ext)]
    with tqdm_joblib(tqdm(desc="Converting files...", total=len(files))) as progress_bar:
        Parallel(n_jobs=n_jobs)(delayed(process_file)(file, output_folder, output_format) for file in files)

def process_file(file, output_folder, output_format):
    """Process a single file and update the progress bar."""
    convert_audio(file, output_folder, output_format)