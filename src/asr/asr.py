from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio, Dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
import numpy as np

import librosa
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import argparse

from asr_utils import WeightedSeq2SeqTrainer

import warnings 
warnings.filterwarnings("ignore")

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai/whisper-base.en')
    parser.add_argument('--dataset_name_or_path', type=str, default='librispeech')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--max_input_length_in_seconds', type=float, default=30)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/asr/')
    return parser.parse_args()

args = parse_args()

# -------------------------------- Loading dataset --------------------------------
dataset_train = load_dataset("librispeech_asr", "clean", split="train.360")
val_dataset = load_dataset("librispeech_asr", "clean", split="validation")
test_dataset = load_dataset("librispeech_asr", "clean", split="test")

## Split the train dataset into 80% train and 20% holdout to be used in future
dataset_train = dataset_train.train_test_split(test_size=0.2, seed=42)
train_dataset, train_dataset_held_out = dataset_train["train"], dataset_train["test"]

## Remove columns that are not needed
train_dataset = train_dataset.remove_columns(["file", "speaker_id", "id", "chapter_id"])
train_dataset_held_out = train_dataset_held_out.remove_columns(["file", "speaker_id", "id", "chapter_id"])
val_dataset = val_dataset.remove_columns(["file", "speaker_id", "id", "chapter_id"])
test_dataset = test_dataset.remove_columns(["file", "speaker_id", "id", "chapter_id"])

## take the first 1000 samples for training
print("-----------------------")
print("LEN TRAIN DATA: ", len(train_dataset))
print("LEN HELD OUT DATA: ", len(train_dataset_held_out))
print("LEN VALID DATA: ", len(val_dataset))
print("LEN TEST DATA: ", len(test_dataset))
print("-----------------------")

# -------------------------------- Loading model  --------------------------------
print("\n-----------------------------")
print(f"Loading model {args.model_name_or_path}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    args.model_name_or_path, 
    language="English", 
    task="transcribe"
    )
processor = WhisperProcessor.from_pretrained(
    args.model_name_or_path, 
    language="English", 
    task="transcribe"
    )
print("Model loaded.")
print("-----------------------------\n")


# -------------------------------- Preprocessing --------------------------------
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()
augmentation = True
targeted = True

def prepare_dataset(batch):
    ## Load and (possibly) audio data from 48 to 16kHz
    audio = batch["audio"]

    if augmentation:

        ## augment only those audios for which the SubgId is not 0
        if targeted and batch["SubgId"] != 0:

            ## Augmentation:
            # 1: Add noise
            # 2: Change speed up
            # 3: Change pitch
            # 4: Change speed down
            # 5: Add noise + Change speed (up) + Change pitch
            # 6: Add noise + Change speed (down) + Change pitch
            # Augment or not, with a probability of 0.15
            augment = np.random.choice([True, False], p=[0.30, 0.70]) # p=[0.15, 0.85]
            # Choose augmentation type
            augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
            if augment:
                # Choose augmentation type
                augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
                if augmentation_type == 1:
                    # Add noise
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                elif augmentation_type == 2:
                    # Change speed up
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=1.2)
                elif augmentation_type == 3:
                    # Change pitch
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
                elif augmentation_type == 4:
                    # Change speed down
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=0.8)
                elif augmentation_type == 5:
                    # Add noise + Change speed (up) + Change pitch
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=1.2)
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
                elif augmentation_type == 6:
                    # Add noise + Change speed (down) + Change pitch
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=0.8)
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
            
        else: 

            ## Augmentation:
            # 1: Add noise
            # 2: Change speed up
            # 3: Change pitch
            # 4: Change speed down
            # 5: Add noise + Change speed (up) + Change pitch
            # 6: Add noise + Change speed (down) + Change pitch
            # Augment or not, with a probability of 0.15
            augment = np.random.choice([True, False], p=[0.15, 0.85])
            # Choose augmentation type
            augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
            if augment:
                # Choose augmentation type
                augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
                if augmentation_type == 1:
                    # Add noise
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                elif augmentation_type == 2:
                    # Change speed up
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=1.2)
                elif augmentation_type == 3:
                    # Change pitch
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
                elif augmentation_type == 4:
                    # Change speed down
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=0.8)
                elif augmentation_type == 5:
                    # Add noise + Change speed (up) + Change pitch
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=1.2)
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
                elif augmentation_type == 6:
                    # Add noise + Change speed (down) + Change pitch
                    noise = np.random.normal(0, 0.005, audio["array"].shape[0])
                    audio["array"] = audio["array"] + noise
                    audio["array"] = librosa.effects.time_stretch(audio["array"], rate=0.8)
                    audio["array"] = librosa.effects.pitch_shift(audio["array"], sr=audio["sampling_rate"], n_steps=4)
                    

    ## Compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
        ).input_features[0]
    ## Compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    ## Optional pre-processing steps
    transcription = batch["text"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    ## Encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

print("\n-----------------------------")
print("Preprocessing dataset...")
train_dataset = train_dataset.map(
    prepare_dataset, 
    num_proc=8
    )
val_dataset = val_dataset.map(
    prepare_dataset,
    num_proc=8
    )
print("Preprocessing done.")
print("-----------------------------\n")

# -------------------------------- Filtering dataset --------------------------------
def is_audio_in_length_range(length):
    return length < args.max_input_length_in_seconds

print("\n-----------------------------")
print ("Length of train dataset before filtering: ", len(train_dataset), " samples")
train_dataset = train_dataset.filter(is_audio_in_length_range, input_columns=["input_length"])
print("Length of train dataset after filtering: ", len(train_dataset), " samples")

print ("Length of valid dataset before filtering: ", len(val_dataset), " samples")
val_dataset = val_dataset.filter(is_audio_in_length_range, input_columns=["input_length"])
print("Length of valid dataset after filtering: ", len(val_dataset), " samples")
print("-----------------------------\n")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# evaluate with the 'normalised' WER
do_normalize_eval = True

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    ## Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    ## We do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# -------------------------------- Training --------------------------------
print("\n-----------------------------")
print("Training model...")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

total_steps = ((len(dataset_train) // args.batch_size) // args.gradient_accumulation_steps) * args.num_train_epochs

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=max(1, args.batch_size // 4),
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    warmup_steps=500,
    num_train_epochs=args.num_train_epochs,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    )

# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
#     )

trainer = WeightedSeq2SeqTrainer(
    balancing=False,
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    )

processor.save_pretrained(training_args.output_dir)

trainer.train()
print("Training done.")
print("-----------------------------\n")

kwargs = {
    "dataset_tags": args.dataset_name_or_path,
    "dataset": "librispeech",
    "language": "en",
    "model_name": "whisper_base",
    "finetuned_from": args.model_name_or_path,
    "tasks": "automatic-speech-recognition",
    "tags": "whisper,en,asr",
    }

## Get the best model from the training
model = trainer.model
model.save_pretrained(training_args.output_dir + "/best_model", **kwargs)
processor.save_pretrained(training_args.output_dir + "/best_model/", **kwargs)