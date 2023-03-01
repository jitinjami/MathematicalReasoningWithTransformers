import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import (TransformerEncoder, TransformerDecoder,TransformerEncoderLayer, TransformerDecoderLayer)
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1
        
        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2   

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device="cpu"):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device="cpu"):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()
        
        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens
                    
        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                for token in tokens:
                    seq.append(src_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")
            
        return data_list, src_vocab, tgt_vocab, src_max, tgt_max


# DataLoader
def generate_batch(data_batch):
    q_batch, ans_batch = [], []
    for (q_item, ans_item) in data_batch:
        q_batch.append(q_item)
        ans_batch.append(ans_item)
    q_batch = pad_sequence(q_batch)
    ans_batch = pad_sequence(ans_batch)
    return q_batch, ans_batch

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int, nhead: int,
                 dim_feedforward:int = 1024, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz, mydevice):
    mask = (torch.triu(torch.ones((sz, sz), device=mydevice)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, src_vocab, mydevice):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, mydevice)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=mydevice).type(torch.bool)

    src_padding_mask = (src == src_vocab.pad_id).transpose(0, 1)
    tgt_padding_mask = (tgt == src_vocab.pad_id).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train_epoch(model, train_data_loader, optimizer, mydevice, loss_fn):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_data_loader):
        src = src.to(mydevice)
        tgt = tgt.to(mydevice)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, mydevice)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if idx % 10 == 0:
            optimizer.step()
        
        losses += loss.item()
    torch.save(model.state_dict(), 'math1.pth')

    return losses / len(train_data_loader)


def evaluate(model, valid_data_loader, mydevice, loss_fn):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_data_loader)):
        src = src.to(mydevice)
        tgt = tgt.to(mydevice)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, mydevice)
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_data_loader)

def greedy_decode(model, src_vocab, src, src_mask, max_len, start_symbol, mydevice):
    src = src.to(mydevice)
    src_mask = src_mask.to(mydevice)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(mydevice)
    for i in range(max_len-1):
        memory = memory.to(mydevice)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(mydevice).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), mydevice)
                                    .type(torch.bool)).to(mydevice)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == src_vocab.eos_id:
          break
    return ys


def translate(model, src, src_vocab, tgt_vocab, tgt_length):
    model.eval()
    tokens = [src_vocab.string_to_id[i] for i in src]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(transformer, src_vocab, src, src_mask, max_len=tgt_length + 2, start_symbol=src_vocab.sos_id).flatten()
    return "".join([tgt_vocab.id_to_string[int(tok)] for tok in tgt_tokens]).replace("<sos>", "").replace("<eos>", "")

def batch_greedy(model, src_vocab, tgt_vocab, data : DataLoader, num_batches: int = 10):
    score = 0
    for idx, batch in enumerate(data):
        questions = batch[0].T
        answers = batch[1].T
        for jdx,(question,answer) in enumerate(zip(questions,answers)):
            question_text = [src_vocab.id_to_string[int(i)] for i in question]
            question_text = "".join(question_text).replace("<pad>", "")
            answer_text = [tgt_vocab.id_to_string[int(i)] for i in answer]
            answer_text = "".join(answer_text).replace("<sos>", "").replace("<eos>", "").replace("<pad>","")

            soln = translate(model, question_text, src_vocab, tgt_vocab, len(answer_text))
            if (soln == answer_text):
                score += 1
                print(question_text)
                print(soln)
        if idx >= num_batches:
            break

def Accuracy_Computation(model, src_vocab, tgt_vocab, data : DataLoader):
    score = 0
    for idx, batch in enumerate(data):
        questions = batch[0].T
        answers = batch[1].T
        for jdx,(question,answer) in enumerate(zip(questions,answers)):
            question_text = [src_vocab.id_to_string[int(i)] for i in question]
            question_text = "".join(question_text).replace("<pad>", "")
            answer_text = [tgt_vocab.id_to_string[int(i)] for i in answer]
            answer_text = "".join(answer_text).replace("<sos>", "").replace("<eos>", "").replace("<pad>","")

            soln = translate(model, question_text, src_vocab, tgt_vocab, len(answer_text))
            if (soln == answer_text):
                score += 1
        if idx >= 100:
            break
    accuracy = 100 * score/(jdx+1)
    return accuracy
# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...
cwd = os.getcwd()
DATASET_DIR = cwd+"/content"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = "numbers__place_value"
# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False)

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False, collate_fn=generate_batch)


SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 2
NUM_EPOCHS = 20

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, NHEAD,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
transformer = transformer.to(mydevice)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

q1 = "What is the ten thousands digit of 62795675?"
a1 = '9'
q2 = "What is the hundred thousands digit of 82923295?"
a2 = '9'
q3 = "What is the tens digit of 70750657?"
a3 = '5'
qs = [q1, q2 ,q3]
a_s = [a1, a2, a3]

for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_data_loader, optimizer, mydevice, loss_fn)
    end_time = time.time()
    val_loss = evaluate(transformer, valid_data_loader, mydevice, loss_fn)
    accu_train = Accuracy_Computation(transformer, src_vocab, tgt_vocab, train_data_loader)
    accu_valid = Accuracy_Computation(transformer, src_vocab, tgt_vocab, valid_data_loader)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}, "
            f"Epoch time = {(end_time - start_time):.8f}s, Training Accuracy: {accu_train}, Validation Accuracy: {accu_valid}"))
    if epoch % 5 == 0:
        for q,a in zip(qs,a_s):
            print("Question: " + q)
            print(translate(transformer, q, src_vocab, tgt_vocab, tgt_size=len(a)))

#transformer.load_state_dict(torch.load('runs1/math1.pth', map_location=torch.device('cpu') ))

accu_train = Accuracy_Computation(transformer, src_vocab, tgt_vocab, train_data_loader)
accu_valid = Accuracy_Computation(transformer, src_vocab, tgt_vocab, valid_data_loader)
print(f"Training Accuracy: {accu_train}, Validation Accuracy: {accu_valid}")

print("Final predictions")
for q,a in zip(qs,a_s):
    print("Question: " + q)
    print(translate(transformer, q, src_vocab, tgt_vocab, len(a)))
