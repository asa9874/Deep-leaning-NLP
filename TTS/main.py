import os
import torch
import requests
import urllib.parse
import pyaudio
import wave
import sys
import os


# https://github.com/ardha27/AI-Waifu-Vtuber
# VoceVox 프로그램 키고 해야함
def voicevox_tts():
    voicevox_url = 'http://localhost:50021'
    katakana_text = "マクドナルドでハンバーガーを買いました。"
    params_encoded = urllib.parse.urlencode({'text': katakana_text, 'speaker': 10})
    request = requests.post(f'{voicevox_url}/audio_query?{params_encoded}')
    params_encoded = urllib.parse.urlencode({'speaker': 46, 'enable_interrogative_upspeak': True})
    request = requests.post(f'{voicevox_url}/synthesis?{params_encoded}', json=request.json())

    with open("test.wav", "wb") as outfile:
        outfile.write(request.content)


def play_wav():
    chunk = 1024  
    path = 'test.wav'
    with wave.open(path, 'rb') as f:
        p = pyaudio.PyAudio()  
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)
                        
        data = f.readframes(chunk)  
        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  

        stream.stop_stream()  
        stream.close()  
        p.terminate()

if __name__ == "__main__":
    voicevox_tts()
    play_wav()
    os.remove('test.wav')
    
   