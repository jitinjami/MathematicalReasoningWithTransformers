## MathematicalReasoningWithTransformers

This project is about implementing an encoder-decoder based [Transformer](https://arxiv.org/abs/1706.03762) using PyTorch to solve basic mathematical problems that are subsets of the [DeepMind mathematical dataset](https://github.com/deepmind/mathematics_dataset).

We treat this problem as a character-level sequence-to-sequence mapping problem: the encoder reads the question as the input sequence, and the decoder outputs the answer sequence

## Motivation
This project was a part of Assignment 4 of the "Deep Learning Lab" course at USI, Lugano taken by [Dr. Kazuki Irie](https://people.idsia.ch/~kazuki/).

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

Experiments were run on 2 modules:
- Compare - Sort
- Numbers - Place value

## Class descriptions
1. `Vocabulary`

This is the class that defines the vocabulary of the text. It tokenizes the unique characters in your text. It has methods to search for the token of a character or create new ones if its a new unique character.

2. `ParallelTextDataset`

It prepares the text from source and target files to be tokenized with the help of the `Vocabulary` class that can be used by the Transformer model.

3. `PositionalEncoding`

To generate positional encoding for the source and target dataset before encoder and decoder respectively.

4. `Seq2SeqTransformer`

The main transformer model with positional encoding, embedding, encoder, decoder and output linear layer.