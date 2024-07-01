from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from pydub import AudioSegment
from datasets import Dataset, Features, Value, Audio
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from datasets import load_metric
import wandb, os
import numpy as np

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
#Prepare dataset
def process_audio(row):
    audio_path = row['path']
    processed_audio_path = f'datasets/originalwavs/{audio_path}.wav'
    #print(processed_audio_path)
    return processed_audio_path


# Load your TSV data into a pandas DataFrame
test_tsv_file_path = 'datasets/original/test.tsv'
train_tsv_file_path = 'datasets/original/train.tsv'

test_df = pd.read_csv(test_tsv_file_path, delimiter='\t')
train_df = pd.read_csv(train_tsv_file_path, delimiter='\t')

test_df['path'] = test_df.apply(process_audio, axis=1)
train_df['path'] = train_df.apply(process_audio, axis=1)

# Define the new features based on the transformed data
new_features = Features({
    'sentence': Value(dtype='string', id=None),
    'path': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
})

# Create a new dataset with the updated features
common_voice_test = Dataset.from_pandas(test_df, features=new_features)
common_voice_train = Dataset.from_pandas(train_df, features=new_features)

print(common_voice_train["path"][0])

import re
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\—\⅛\–\']'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
    
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
    

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#print(common_voice_train[0]["audio"])

#common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
#common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

#print(common_voice_train[0]["audio"])
wandb.login()
wandb_project = "name_project"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    
from datetime import datetime

project = "finetune"
base_model_name = "wav2vec"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

def prepare_dataset(batch):
    audio = batch["path"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="wav2vec_results",
  group_by_length=True,
  per_device_train_batch_size=2,
  gradient_accumulation_steps=1,
  evaluation_strategy="steps",
  max_steps=4000,
  gradient_checkpointing=True,
  fp16=False,
  save_steps=1000,
  eval_steps=1000,
  logging_steps=1000,
  learning_rate=1e-5,
  warmup_steps=500,
  save_total_limit=5,
  push_to_hub=False,
  report_to="wandb",
  run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}" # testing
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')



