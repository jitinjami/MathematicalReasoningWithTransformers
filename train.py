import argparse
import os
import time
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.defaults import _C as cfg
from src.text_dataset import ParallelTextDataset
from src.utils import generate_batch, train_epoch, evaluate, Accuracy_Computation, translate
from src.model import Seq2SeqTransformer

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify src.defaults")
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    cfg.merge_from_list(args.opts) #'defaults.py' has some default settings, if they have been changed by args, that gets consolidated here
    #cfg.freeze()

    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwd = os.getcwd()
    cfg.DATASET_DIR = cwd + "/data"
    cfg.MODEL_SAVE_PATH = cwd + "/results/"

    src_file_path = f"{cfg.DATASET_DIR}/{cfg.TASK}/{cfg.TRAIN_FILE_NAME}{cfg.INPUTS_FILE_ENDING}"
    tgt_file_path = f"{cfg.DATASET_DIR}/{cfg.TASK}/{cfg.TRAIN_FILE_NAME}{cfg.TARGETS_FILE_ENDING}"

    train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

    #get the vocab
    src_vocab = train_set.src_vocab
    tgt_vocab = train_set.tgt_vocab

    print(type(src_vocab))
    #save the vocab
    with open(cfg.MODEL_SAVE_PATH + "src_vocab.pickle", "wb") as outfile:
        pickle.dump(src_vocab, outfile)
    
    with open(cfg.MODEL_SAVE_PATH + "tgt_vocab.pickle", "wb") as outfile:
        pickle.dump(tgt_vocab, outfile)
    
    print("Saved vocab files")

    src_file_path = f"{cfg.DATASET_DIR}/{cfg.TASK}/{cfg.VALID_FILE_NAME}{cfg.INPUTS_FILE_ENDING}"
    tgt_file_path = f"{cfg.DATASET_DIR}/{cfg.TASK}/{cfg.VALID_FILE_NAME}{cfg.TARGETS_FILE_ENDING}"

    valid_set = ParallelTextDataset(
        src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        extend_vocab=False)

    batch_size = 64

    train_data_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    valid_data_loader = DataLoader(
        dataset=valid_set, batch_size=batch_size, shuffle=False, collate_fn=generate_batch)
    
    cfg.SRC_VOCAB_SIZE = len(src_vocab)
    cfg.TGT_VOCAB_SIZE = len(tgt_vocab)

    
    transformer = Seq2SeqTransformer(cfg.NUM_ENCODER_LAYERS, cfg.NUM_DECODER_LAYERS,
                                 cfg.EMB_SIZE, cfg.SRC_VOCAB_SIZE, cfg.TGT_VOCAB_SIZE, cfg.NHEAD,
                                 cfg.FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(mydevice)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_vocab.pad_id)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    for epoch in range(1, cfg.NUM_EPOCHS+1):
        start_time = time.time()
        train_loss = train_epoch(transformer, train_data_loader, src_vocab, optimizer, mydevice, loss_fn, cfg.MODEL_SAVE_PATH + cfg.TASK + ".pth")
        end_time = time.time()
        val_loss = evaluate(transformer, valid_data_loader, src_vocab, mydevice, loss_fn)
        accu_train = Accuracy_Computation(transformer, src_vocab, tgt_vocab, mydevice, train_data_loader)
        accu_valid = Accuracy_Computation(transformer, src_vocab, tgt_vocab, mydevice, valid_data_loader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}, "
                f"Epoch time = {(end_time - start_time):.8f}s, Training Accuracy: {accu_train}, Validation Accuracy: {accu_valid}"))

     
    accu_train = Accuracy_Computation(transformer, src_vocab, tgt_vocab, mydevice, train_data_loader)
    accu_valid = Accuracy_Computation(transformer, src_vocab, tgt_vocab, mydevice, valid_data_loader)
    print(f"Training Accuracy: {accu_train}, Validation Accuracy: {accu_valid}")


if __name__ == '__main__':
    main()