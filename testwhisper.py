import queue
import sounddevice as sd
import numpy as np
import whisper
import wave
import tempfile
import threading

def record_audio_stop_on_enter():
    """Graba audio del micrófono hasta que se presione Enter."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    print("Presiona Enter para detener la grabación...")
    q = queue.Queue()

    # Iniciar la grabación en un hilo separado
    stream = sd.InputStream(callback=callback, dtype='int16', channels=1, samplerate=16000)
    with stream:
        threading.Thread(target=stream.start).start()
        input()  # Espera a que el usuario presione Enter
        stream.stop()

    # Recolectar datos del audio
    recording = []
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        recording.append(data)

    recording = np.concatenate(recording, axis=0)
    return recording

def save_wav(file_path, data, samplerate=16000):
    """Guarda los datos de numpy array a un archivo WAV."""
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # los samples son de int16, por lo tanto 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(data)

def transcribe_audio(file_path):
    """Transcribe el audio a texto utilizando Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

# Proceso de grabación y transcripción
audio_data = record_audio_stop_on_enter()
temp_file = tempfile.mktemp(".wav")  # Crea un archivo temporal
save_wav(temp_file, audio_data, 16000)  # Guarda la grabación en un archivo WAV

# Usar Whisper para transcribir el audio
transcription = transcribe_audio(temp_file)
print("Transcripción:", transcription)
