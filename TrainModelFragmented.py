from datasets import Dataset, concatenate_datasets, load_from_disk
import pandas as pd
from datasets import Audio
import gc
import os
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate

# Cargar datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.columns = ["audio", "sentence"]
test_df.columns = ["audio", "sentence"]

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Cargar procesadores
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Spanish", task="transcribe")

def prepare_dataset(examples):
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    sentences = examples["sentence"]
    examples["labels"] = tokenizer(sentences).input_ids
    return examples

# Dividir y procesar el dataset en fragmentos más pequeños y guardarlos en disco
batch_size = 500

def process_in_batches(dataset, prepare_function, save_path):
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

# Cargar los datasets procesados desde el disco
def load_temp_datasets(temp_paths):
    datasets = [load_from_disk(path) for path in temp_paths]
    return concatenate_datasets(datasets)

train_dataset = load_temp_datasets(train_temp_datasets)
test_dataset = load_temp_datasets(test_temp_datasets)

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