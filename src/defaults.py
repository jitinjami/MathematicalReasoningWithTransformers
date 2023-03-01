from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN_FILE_NAME = "train"
_C.VALID_FILE_NAME = "interpolate"

_C.INPUTS_FILE_ENDING = ".x"
_C.TARGETS_FILE_ENDING = ".y"

_C.TASK = "numbers__place_value"
# _C.TASK = "comparison__sort"
# _C.TASK = "algebra__linear_1d"

_C.EMB_SIZE = 256
_C.NHEAD = 8
_C.FFN_HID_DIM = 1024
_C.BATCH_SIZE = 64
_C.NUM_ENCODER_LAYERS = 3
_C.NUM_DECODER_LAYERS = 2
_C.NUM_EPOCHS = 20
