# Note: The core logic previously in this file (atom, plugin, decompose, etc.)
# and associated global state has been refactored into the AtomProcessor class
# located in experiment/processor.py.
# This file is kept for potential future use or if other parts of the project
# still import from it directly, but its primary functionality is now deprecated.

# Imports that might still be used elsewhere (or were used by the moved functions)
from functools import wraps
from experiment.utils import (
	extract_json,
	extract_xml,
	calculate_depth,
	score_math,
	score_mc,
	score_mh,
)
from llm import gen
from experiment.prompter import math, multichoice, multihop
from contextlib import contextmanager
import asyncio

# --- Global variables and functions removed ---
# count, MAX_RETRIES, LABEL_RETRIES, ATOM_DEPTH, score, module, prompter
# set_module, retry, decompose, merging, atom, plugin, direct, multistep,
# label, contract, ensemble, temporary_retries
# --- All moved to experiment/processor.py -> AtomProcessor class ---