# Lint as: python3
r"""Quick-start demo for a sentiment analysis model.

This demo fine-tunes a small Transformer (BERT-tiny) on the Stanford Sentiment
Treebank (SST-2), and starts a LIT server.

To run locally:
  python -m lit_nlp.examples.quickstart_sst_demo \
      --port=5432

Training should take less than 5 minutes on a single GPU. Once you see the
ASCII-art LIT logo, navigate to localhost:5432 to access the demo UI.
"""
import tempfile
import os
import json
import pandas as pd
import re
import random

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

from typing import Optional, Dict, List, Iterable

from absl import logging
import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf
import transformers

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec

import tensorflow_datasets as tfds

# NOTE: additional flags defined in server_flags.py


# OLD IMPORT START ----------------------------------------------------------------
# from the previous notebooks - creating interrupted utternaces

def preload_data():
    # from MultiWOZ-Parser; reading the names of the files for training, testing, validation datasets
    # https://github.com/jojonki/MultiWOZ-Parser/blob/master/parser.py

    def load_json(data_file):
        if os.path.isfile(data_file):
            with open(data_file, 'r') as read_file:
                data = json.load(read_file)
                return data

    def load_list_file(list_file):
        with open(list_file, 'r') as read_file:
            dialog_id_list = read_file.readlines()
            dialog_id_list = [l.strip('\n') for l in dialog_id_list]
            return dialog_id_list
        return
    
    # extracts the utterances from the MultiWOZ dataset
    def get_utterances(data):
        utterances = []

        for block in data:
            data = block['log']

            for ut in data:
                # replacing whitespace characters with spaces
                text = re.sub("\\s", " ", ut['text'])
                text = re.sub("[^a-zA-Z0-9 ]+", "", ut['text'])

                utterances.append(text)

        return utterances
    
    def split_data(data):
        X = []
        Y = []
        for i in range(len(data)):
            tokens = data[i].split()

            if (i <= len(data)/2) and (len(tokens) > 4):
                # picking random point for splitting the conversation turn
                l = random.randrange(1, len(tokens) - 3)
                # splitting data
                X.append(' '.join(tokens[:l]))
                # adding 0 to the target list -> 0 -- interrupted turn 
                Y.append('interrupted')

            # second section of the dataset is made out of full utterances
            else:
                # adding the full uninterrupted conversation turn
                X.append(data[i])
                # adding 1 to the target list -> 1 -- uninterrupted turn 
                Y.append('finished')

        # shuffling the dataset
        c = list(zip(X, Y, data))
        random.shuffle(c)
        X, Y, data = zip(*c)

        return X,Y,data

    # extracting data
    dialog_data_file = './datasets/multiwoz/data.json'
    dialog_data = load_json(dialog_data_file)
    dialog_id_list = list(set(dialog_data.keys()))

    valid_list_file = './datasets/multiwoz/valListFile.json'
    test_list_file = './datasets/multiwoz/testListFile.json'

    valid_id_list = list(set(load_list_file(valid_list_file)))
    test_id_list = load_list_file(test_list_file)
    train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]

    train_data = [v for k, v in dialog_data.items() if k in train_id_list]
    valid_data = [v for k, v in dialog_data.items() if k in valid_id_list]
    test_data = [v for k, v in dialog_data.items() if k in test_id_list]
    
    # merging all datasets together
    data = train_data + valid_data + test_data
    utterances = get_utterances(data)
    
    X, Y, data_clean = split_data(utterances)
    
    return pd.DataFrame(data={'text': X, 'label': Y}).sample(20000)


# OLD IMPORT END ----------------------------------------------------------------

# MODELS FILE -------------------------------------------------------------------

class UEDModel(glue_models.GlueModel):
  """Classification model on SST-2."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name='text',
        text_b_name=None,
        labels = ['finished',
                  'interrupted'],
        null_label_idx=0,
        **kw)

# DATASET FILE ---------------------------------------------------------------

class UEDData(lit_dataset.Dataset):
    LABELS = ['finished', 'interrupted']

#     LABELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    def __init__(self, split: str):
        df = preload_data()
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{
          'text': row['text'],
          'label': row['label'],
        } for _, row in df.iterrows()]

    def spec(self):
        return {
            'text': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.LABELS)
        }

# MAIN FILE ------------------------------------------------------------------

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "encoder_name", "google/bert_uncased_L-2_H-128_A-2",
    "Encoder name to use for fine-tuning. See https://huggingface.co/models.")

flags.DEFINE_string("model_path", None, "Path to save trained model.")


def run_finetuning(train_path):
    """Fine-tune a transformer model."""
    train_data = UEDData("train")
    val_data = UEDData("validation")
    model = UEDModel(FLAGS.encoder_name, for_training=True)
    model.train(train_data.examples, validation_inputs=val_data.examples)
    model.save(train_path)

def main(_):
    model_path = FLAGS.model_path or tempfile.mkdtemp()
    logging.info("Working directory: %s", model_path)
    run_finetuning(model_path)

    # Load our trained model.
    models = {"sst": UEDModel(model_path)}
    datasets = {"sst_dev": UEDData("validation")}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
