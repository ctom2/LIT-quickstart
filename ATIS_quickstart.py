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

# MY IMPORT ------------------------------------------------------------------

def load_json(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            lines = read_file.readlines()
            return [l.strip('\n').lstrip() for l in lines]
        
        
def get_dataframe(data):
    texts = []
    intents = []
    
    for line in data:
        if 'text' in line:
            texts.append(line[len("text\":  \""):-2])
        elif 'intent' in line:
            i = line[len("intent\":  \""):-2]

            # it there are more intents, only the first one is selected
            if '+' in i: i = i.split('+')[0]
            
            intents.append(i)
    
    return pd.DataFrame({'text':texts, 'label':intents})

# MODELS FILE ----------------------------------------------------------------

class ATISModel(glue_models.GlueModel):
  """Classification model on SST-2."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name="text",
        text_b_name=None,
        labels = ['flight',
                  'airfare',
                  'ground_service',
                  'airline',
                  'abbreviation',
                  'aircraft',
                  'flight_time',
                  'quantity',
                  'airport',
                  'distance',
                  'city',
                  'ground_fare',
                  'capacity',
                  'flight_no',
                  'restriction',
                  'meal',
                  'cheapest'],

#         labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
        
        null_label_idx=0,
        **kw)

# DATASET FILE ---------------------------------------------------------------

class ATISData(lit_dataset.Dataset):
    LABELS = ['flight',
              'airfare',
              'ground_service',
              'airline',
              'abbreviation',
              'aircraft',
              'flight_time',
              'quantity',
              'airport',
              'distance',
              'city',
              'ground_fare',
              'capacity',
              'flight_no',
              'restriction',
              'meal',
              'cheapest']

    def __init__(self, split: str):
        df = get_dataframe(load_json('./datasets/atis/train.json'))
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
    train_data = ATISData("train")
    val_data = ATISData("validation")
    model = ATISModel(FLAGS.encoder_name, for_training=True)
    model.train(train_data.examples, validation_inputs=val_data.examples)
    model.save(train_path)

def main(_):
    model_path = FLAGS.model_path or tempfile.mkdtemp()
    logging.info("Working directory: %s", model_path)
    run_finetuning(model_path)

    # Load our trained model.
    models = {"sst": ATISModel(model_path)}
    datasets = {"sst_dev": ATISData("validation")}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

