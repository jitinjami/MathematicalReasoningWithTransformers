# MathematicalReasoningWithTransformers

This project is about implementing an encoder-decoder based [Transformer](https://arxiv.org/abs/1706.03762) using PyTorch to solve basic mathematical problems that are subsets of the [DeepMind mathematical dataset](https://github.com/deepmind/mathematics_dataset).

We treat this problem as a character-level sequence-to-sequence mapping problem: the encoder reads the question as the input sequence, and the decoder outputs the answer sequence

## Tech used
<b>Built with</b>
- [Python3](https://www.python.org)
- [NumPy](https://numpy.org)
- [PyTorch](https://pytorch.org)

## Features
The project includes implementation of the following concepts:
- Embedding layers
- Transformer encoder and decoder modules
- Positional encodings
- Greedy search with stopping criteria and batch mode evaluation
- Managing gradient accumulation
- Hyper-parameter tuning with `NHEAD` (number of attention heads) and `FFN_HID_DIM` (Feed Forward Dimensions)

Experiments were run on module:
- Numbers - Place value

## Data Prelimnaries
We solve basic mathematical questions of the `Numbers - Place value` module in the [DeepMind mathematical dataset](https://github.com/deepmind/mathematics_dataset). The questions are in the form:

_Question_: What is the hundred millions digit of 217883211? \
_Answer_: 2

The preprocessed dataset can be found in [Numbers - Place value](./data/numbers__place_value/) folder. The module consists of `train.x` (_questions_) and `train.y` (_answers_) for training and `interpolate.x` (_questions_) and `interpolate.y` (_answers_) for validation. Here is quick run down of the dataset properties:


| Syntax      | Description | Description |
| ----------- | ----------- | ----------- |
| Character vocabulary size      | 33       | 33       |
| Number of questions   | 1999998        | 10000        |
| Number of characters      | 63898109       | 339578       |
| Average length of question   | 32        | 34        |
| Total number of words      | 14531838       | 73641       |

## Vocabulary and Dataloader
The implementation of [Vocabulary](./src/vocabulary.py) class helps define the vocabulary of the text. The source and target material get different vocabulary instances but are built from the training data. The validation data uses the same vocabulary instances for its source and target material

The [ParallelTextDataset](./src/text_dataset.py) class prepares the text from source and target files to be tokenized with the help of the `Vocabulary` class that can be used by the Transformer model.

## Model
A [Seq2SeqTransformer](./src/model.py) model was implemented with the following features:
- Embedding layers
- Transformer encoder and decoder modules
- Positional encoding

The transformer is initiated with [default configuration](./config/defaults.py) and dataset properties. A source and target mask is created using [create_mask](./src/utils.py#L20) before training.

## Greedy Search
We implement greedy search using [greedy_decode](./src/utils.py#L76). 

In the greedy search algorithm, we use `encode` method of a trained transformer to get `memory` out of the `source` (which is the essentially the question). We use `memory` and the `decode` function to get the first character of our `target` (the character token with the highest probability amongst the tokens in target vocab). We then use this first character and `memory` to get the next character and so on while we concatenate the prediction to the previous prediction.

We use the `encode` function to learn the context vector which is then used by the `decode` function to generate the logits. Once the probability distribution was present, implementing greedy search was straightforward where we pick the character with the highest probability in the vocabulary.

A stopping mechanism was implemented to terminate greedy search once the `<eos>` tag is pre- dicted. The termination of search once we reach maximum length is implemented in the [translate](./src/utils.py#L98) function, where the prediction length is capped at maximum target length with the flag `max_len`.

## Batch Greedy Search
A batch mode evaluation implementation was carried out in the [batch_greedy](./src/utils.py#L107) function where the greedy search is carried out batch by batch. The evaluation moves to the next batch only once we finish all the questions in this batch. There is also an argument to set the number of batches you want to carry out greedy search on, by default this is set to 10. The function also prints the question and prediction for predictions that are correct.

## Demo
A demo can be found in the [demo.ipynb](./demo.ipynb) jupyter notebook.