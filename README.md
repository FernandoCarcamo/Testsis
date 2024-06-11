# Proyecto de Tesis: Fine-tuning del Modelo Whisper para Transcripción en Español

Este proyecto se centra en el entrenamiento y ajuste fino del modelo Whisper de OpenAI para la tarea de transcripción de audio en español. Todo el proyecto está desarrollado en Python.

## Descripción

El objetivo de este proyecto es ajustar el modelo Whisper para transcribir audio en español. Utiliza datasets de entrenamiento y prueba que contienen pares de audio y texto.

## Requisitos

- Python 3.8 o superior
- pip (Gestor de paquetes de Python)

## Instalación

1. **Clonar el repositorio**
    ```sh
    git clone "Repo"
    cd "repo"
    ```

2. **Descargar y preparar los datasets**

    Coloca los archivos `train.csv` y `test.csv` en el directorio del proyecto.

## Dependencias

Asegúrate de tener instaladas las siguientes dependencias en tu entorno de Python:

- `transformers`
- `datasets`
- `evaluate`
- `pandas`
- `torch`
- `soundfile`
- `collections`

Estas dependencias se pueden instalar mediante el archivo `requirements.txt` mencionado en la sección de instalación.

## Uso

### Entrenamiento del Modelo

1. **Cargar los datasets**
    ```python
    import pandas as pd
    from datasets import Dataset, Audio

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    ```

2. **Preparar los datasets**
    ```python
    train_df.columns = ["audio", "sentence"]
    test_df.columns = ["audio", "sentence"]
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    ```

3. **Configurar los procesadores**
    ```python
    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")
    ```

4. **Preparar los datos**
    ```python
    def prepare_dataset(examples):
        audio = examples["audio"]
        examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        sentences = examples["sentence"]
        examples["labels"] = tokenizer(sentences).input_ids
        return examples

    batch_size = 500

    def process_in_batches(dataset, prepare_function, save_path):
        import os
        import gc
        os.makedirs(save_path, exist_ok=True)
        temp_datasets = []
        for i in range(0, len(dataset), batch_size):
            subset = dataset.select(range(i, min(i + batch_size, len(dataset))))
            processed_subset = subset.map(prepare_function, num_proc=1)
            temp_path = os.path.join(save_path, f"batch_{i // batch_size}")
            processed_subset.save_to_disk(temp_path)
            temp_datasets.append(temp_path)
            del subset, processed_subset
            gc.collect()
        return temp_datasets

    train_save_path = "train_temp"
    test_save_path = "test_temp"

    train_temp_datasets = process_in_batches(train_dataset, prepare_dataset, train_save_path)
    test_temp_datasets = process_in_batches(test_dataset, prepare_dataset, test_save_path)
    ```

5. **Cargar los datasets procesados**
    ```python
    from datasets import load_from_disk, concatenate_datasets

    def load_temp_datasets(temp_paths):
        datasets = [load_from_disk(path) for path in temp_paths]
        return concatenate_datasets(datasets)

    train_dataset = load_temp_datasets(train_temp_datasets)
    test_dataset = load_temp_datasets(test_temp_datasets)
    ```

6. **Definir el `DataCollator` y las métricas**
    ```python
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import torch
    import evaluate

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    ```

7. **Configurar y entrenar el modelo**
    ```python
    from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.language = "spanish"
    model.config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned-es",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25,
        eval_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    ```

### Transcripción de Audio utilizando los datos nuevos

1. **Cargar las rutas de los checkpoints**
    ```python
    checkpoint_paths = [
        r"Path correspondiente",
        r"Path correspondiente",
        r"Path correspondiente",
        r"Path correspondiente"
    ]
    ```

2. **Cargar el procesador original**
    ```python
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    ```

3. **Definir la función de transcripción**
    ```python
    import soundfile as sf
    import torch

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
    ```

4. **Ejemplo de uso con un nuevo archivo de audio**
    ```python
    new_audio_path = r"Path correspondiente"
    ```

5. **Obtener transcripciones de cada checkpoint y almacenarlas con sus puntuaciones de confianza**
    ```python
    from transformers import WhisperForConditionalGeneration

    transcriptions_with_confidences = []

    for checkpoint_path in checkpoint_paths:
        # Cargar el modelo finetuned desde la carpeta del checkpoint
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        print(f"Modelo cargado desde {checkpoint_path}")

        # Obtener la transcripción y la puntuación de confianza
        transcription, confidence_score = transcribe_audio(model, new_audio_path)
        transcriptions_with_confidences.append((transcription, confidence_score))
        print(f"Transcription from {checkpoint_path}: {transcription} with confidence score {confidence_score}")
    ```

6. **Combinar las transcripciones utilizando una estrategia de votación ponderada**
    ```python
    from collections import Counter

    combined_transcriptions = Counter()

    for transcription, confidence_score in transcriptions_with_confidences:
        combined_transcriptions[transcription] += confidence_score

    # Seleccionar la transcripción con la puntuación más alta
    best_transcription = combined_transcriptions.most_common(1)[0][0]

    # Mostrar el resultado final
    print("Final transcription result:", best_transcription)
    ```
## Dataset Utilizado

El dataset utilizado para este proyecto es el TEDx Spanish Corpus, que se puede descargar desde el siguiente [enlace](https://www.openslr.org/67/).

**Información del dataset:**

- **Identificador**: SLR67
- **Resumen**: Datos en español tomados de las charlas TEDx.
- **Categoría**: Speech
- **Licencia**: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
- **Descargas**: 
  - [tedx_spanish_corpus.tgz (2.3G)](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) (Mirrors: [US](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) [EU](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz) [CN](https://www.openslr.org/resources/67/tedx_spanish_corpus.tgz))

**Acerca de este recurso:**

El TEDx Spanish Corpus es un corpus de género desequilibrado de 24 horas de duración. Contiene discursos espontáneos de varios expositores en eventos TEDx; la mayoría son hombres. Las transcripciones se presentan en minúsculas sin signos de puntuación.

El proceso de recopilación de datos fue desarrollado en parte por el programa de servicio social "Desarrollo de Tecnologías del Habla" que depende de la Universidad Nacional Autónoma de México y en parte por el proyecto CIEMPIESS-UNAM.

Agradecimientos especiales al equipo de TED-Talks por permitirnos compartir este dataset.

**Cita del dataset:**

@misc{mena_2019,
	title = "{TEDx Spanish Corpus. Audio and transcripts in Spanish taken from the TEDx Talks; shared under the CC BY-NC-ND 4.0 license}",
	author = "Hernandez-Mena, Carlos D.",
	howpublished = "Web Download",
	institution = "Universidad Nacional Autonoma de Mexico",
	location = "Mexico City",
	year = "2019"
}

## Configuración del Sistema

Este programa se ejecutó inicialmente en la siguiente configuración de hardware y software:

- **Procesador**: AMD Ryzen 7 3700X 8-Core Processor 3.60 GHz
- **RAM**: 32.0 GB
- **Sistema Operativo**: 64-bit, x64-based processor
- **GPU**: NVIDIA GeForce RTX 3060
- **Almacenamiento**: 1.5 TB
- **Sistema operativo**: Windows

## Contribución

Si deseas contribuir a este proyecto, por favor sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Agrega nueva funcionalidad'`).
4. Sube los cambios a tu rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Para más información, consulta el archivo `LICENSE`.
