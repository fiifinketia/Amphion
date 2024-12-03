# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import torchaudio
from utils import audio
import csv
import random
from datasets import load_dataset, Audio
import requests
from utils.util import has_existed
from text import _clean_text
import librosa
import soundfile as sf
from scipy.io import wavfile

from pathlib import Path
import numpy as np


def get_uid2utt(dataset, cfg):
    index_count = 0
    total_duration = 0

    uid2utt = []
    for example in dataset:

        res = {
            "Dataset": "UGSpeechAT",
            "index": index_count,
            "Singer": "UGSpeech",
            "Uid": example["audio"].split("/")[-1].replace(".wav", ""),
            "Text": example["transcription"],
        }

        # Duration in wav files
        target_dst = f"~/tmp/{example['audio'].split('/')[-1]}"
        torchaudio.utils.download.download_url_to_file(example["audio"], target_dst)

        res["Path"] = target_dst

        waveform, sample_rate = torchaudio.load(target_dst)
        duration = waveform.size(-1) / sample_rate
        res["Duration"] = duration

        uid2utt.append(res)

        index_count = index_count + 1
        total_duration += duration
        print(res)
        print(total_duration)

    return uid2utt, total_duration / 3600


def split_dataset(
    lines, test_rate=0.05, valid_rate=0.05, test_size=None, valid_size=None
):
    if test_size == None:
        test_size = int(len(lines) * test_rate)
    if valid_size == None:
        valid_size = int(len(lines) * valid_rate)
    random.shuffle(lines)

    train_set = []
    test_set = []
    valid_set = []

    for line in lines[:test_size]:
        test_set.append(line)
    for line in lines[test_size : test_size + valid_size]:
        valid_set.append(line)
    for line in lines[test_size + valid_size :]:
        train_set.append(line)
    return train_set, test_set, valid_set


max_wav_value = 32768.0


def main(output_path, dataset_path, cfg):
    print("-" * 10)
    print("Dataset splits for {}...\n".format("UGSpeech Audio Transcribed"))

    dataset = "UGSpeechAT"

    save_dir = os.path.join(output_path, dataset)
    os.makedirs(save_dir, exist_ok=True)

    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")

    speaker = "UGSpeech"
    speakers = [dataset + "_" + speaker]
    singer_lut = {name: i for i, name in enumerate(sorted(speakers))}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)

    if (
        has_existed(train_output_file)
        and has_existed(test_output_file)
        and has_existed(valid_output_file)
    ):
        return

    dataset = load_dataset("fiifinketia/ugspeech-tts-akan", split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    res, hours = get_uid2utt(dataset["train"], cfg)

    # Save train
    os.makedirs(save_dir, exist_ok=True)
    with open(train_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Train_hours= {}".format(hours))

    res, hours = get_uid2utt(dataset["test"], cfg)

    # Save test
    os.makedirs(save_dir, exist_ok=True)
    with open(test_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Test_hours= {}".format(hours))

    # Save valid
    os.makedirs(save_dir, exist_ok=True)
    with open(valid_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Valid_hours= {}".format(hours))
