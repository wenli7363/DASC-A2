# test_fix.py
import seqeval
print("seqeval version:", seqeval.__version__)
from seqeval.metrics import f1_score
print("Test f1_score:", f1_score([['O', 'B-PER']], [['O', 'B-PER']]))