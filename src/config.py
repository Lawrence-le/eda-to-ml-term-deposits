# config.py

import os

# Get the absolute path of the project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "../data/bmarket.db")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "../outputs/")

IMPORTANT_FEATURES = [
    "Age",
    "Occupation",
    "Campaign Calls",
    "Previous Contact Days",
    "Credit Default",
    "Subscription Status",
]
