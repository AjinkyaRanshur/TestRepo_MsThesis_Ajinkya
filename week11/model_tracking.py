"""
Model Manager - Centralized model tracking

It uses a json file to track all your models and their progress so far will return the info to interface for user to choose from

"""
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List




