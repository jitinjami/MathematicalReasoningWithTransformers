from src.utils import translate
from src.defaults import _C as cfg
import argparse
import os
import pickle
from src.model import Seq2SeqTransformer
import torch

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify src.defaults")
    args = parser.parse_args()
    return args

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.getcwd()

args = get_args()
cfg.merge_from_list(args.opts)

cfg.DATASET_DIR = cwd + "/data"
cfg.MODEL_SAVE_PATH = cwd + "/results/"

with open(cfg.MODEL_SAVE_PATH + "src_vocab.pickle", "rb") as infile:
    src_vocab = pickle.load(infile)
    
with open(cfg.MODEL_SAVE_PATH + "tgt_vocab.pickle", "rb") as infile:
    tgt_vocab = pickle.load(infile)

cfg.SRC_VOCAB_SIZE = len(src_vocab)
cfg.TGT_VOCAB_SIZE = len(tgt_vocab)

transformer = Seq2SeqTransformer(cfg.NUM_ENCODER_LAYERS, cfg.NUM_DECODER_LAYERS,
                                 cfg.EMB_SIZE, cfg.SRC_VOCAB_SIZE, cfg.TGT_VOCAB_SIZE, cfg.NHEAD,
                                 cfg.FFN_HID_DIM)


transformer.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH + cfg.TASK + ".pth", map_location=mydevice))

q1 = "What is the ten thousands digit of 62795675?"
a1 = '9'
q2 = "What is the hundred thousands digit of 82923295?"
a2 = '9'
q3 = "What is the tens digit of 70750657?"
a3 = '5'
qs = [q1, q2 ,q3]
a_s = [a1, a2, a3]

print("Final predictions")
for q,a in zip(qs,a_s):
    print("Question: " + q)
    print(translate(transformer, q, src_vocab, tgt_vocab, len(a), mydevice))
