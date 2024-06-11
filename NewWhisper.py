from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import soundfile as sf
import torch
from collections import Counter
import numpy as np

# Lista de rutas a los checkpoints
checkpoint_paths = [
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-25",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-50",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-75",
    r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\checkpoint-100"
]

# Cargar el procesador original
processor = WhisperProcessor.from_pretrained("openai/whisper-base")


# Función para predecir utilizando un modelo finetuned
def transcribe_audio(model, file_path):
    # Cargar el audio
    audio_input, _ = sf.read(file_path)

    # Procesar el audio
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")

    # Generar la transcripción
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], output_scores=True, return_dict_in_generate=True)
        logits = predicted_ids.scores[0]
        confidences = torch.nn.functional.softmax(logits, dim=-1)
        confidence_score = confidences.max().item()

    # Decodificar la transcripción
    transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
    return transcription, confidence_score


# Ejemplo de uso con un nuevo archivo de audio
new_audio_path = r"C:\Users\renec\PycharmProjects\speechrecognition\TrainignModel\Beta\whisper-finetuned-es\TEDX_F_001_SPA_0001.wav"

# Obtener transcripciones de cada checkpoint y almacenarlas con sus puntuaciones de confianza
transcriptions_with_confidences = []

for checkpoint_path in checkpoint_paths:
    # Cargar el modelo finetuned desde la carpeta del checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    print(f"Modelo cargado desde {checkpoint_path}")

    # Obtener la transcripción y la puntuación de confianza
    transcription, confidence_score = transcribe_audio(model, new_audio_path)
    transcriptions_with_confidences.append((transcription, confidence_score))
    print(f"Transcription from {checkpoint_path}: {transcription} with confidence score {confidence_score}")

# Combinar las transcripciones utilizando una estrategia de votación ponderada
combined_transcriptions = Counter()

for transcription, confidence_score in transcriptions_with_confidences:
    combined_transcriptions[transcription] += confidence_score

# Seleccionar la transcripción con la puntuación más alta
best_transcription = combined_transcriptions.most_common(1)[0][0]

# Mostrar el resultado final
print("Final transcription result:", best_transcription)
