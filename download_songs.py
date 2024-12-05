import os
import yt_dlp
from pydub import AudioSegment

def download_audio_from_youtube(query, output_filename):
    """
    Search for a song on YouTube, download its audio, and save it as an MP3 file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': f"{output_filename}.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
        if 'entries' in search_results and search_results['entries']:
            video_url = search_results['entries'][0]['url']
            ydl.download([video_url])
            print(f"Downloaded MP3 as {output_filename}.mp3")
            return f"{output_filename}.mp3"
        else:
            print("No results found on YouTube.")
            return None

def convert_mp3_to_wav(mp3_file, wav_file):
    """
    Convert an MP3 file to WAV format.
    """
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    audio.export(wav_file, format="wav")
    print(f"Converted {mp3_file} to {wav_file}")
    return wav_file

def get_song_wav(song_name, artist_name):
    """
    Automate the process of searching for a song on YouTube,
    downloading its audio, and converting it to WAV.
    """
    query = f"{song_name} {artist_name}"
    output_filename = f"{song_name} - {artist_name}"
    mp3_file = download_audio_from_youtube(query, output_filename)

    if mp3_file and os.path.exists(mp3_file):
        wav_file = mp3_file.replace('.mp3', '.wav')
        convert_mp3_to_wav(mp3_file, wav_file)
        os.remove(mp3_file)
        return wav_file
    else:
        print("Failed to retrieve audio.")
        return None

# Example Usage
song_name = "Shape of You"
artist_name = "Ed Sheeran"
wav_file = get_song_wav(song_name, artist_name)
if wav_file:
    print(f"WAV file saved as {wav_file}")
