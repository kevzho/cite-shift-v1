"""Step 1: build SciBERT training contexts with the OFFICIAL, unmodified bert.prepare_bert_input()
from THUDM/paper-source-trace (commit d0896c6). Only addition: fixed numpy seed (official code is unseeded)."""
import sys, os, random
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper-source-trace")
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
np.random.seed(42); random.seed(42)
import bert
bert.prepare_bert_input()
