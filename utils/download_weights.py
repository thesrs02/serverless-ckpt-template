#!/usr/bin/env python
import os
import sys
import shutil
from transformers.utils.hub import move_cache


move_cache()
sys.path.append(".")

from src.sd_runner import init, models_cache_dir


if os.path.exists(models_cache_dir):
    shutil.rmtree(models_cache_dir)
os.makedirs(models_cache_dir, exist_ok=True)


init(local_files_only=False)
print("All models downloaded!")
