"""Configuration settings for the K-1 tax form extraction pipeline."""

import os
from typing import Dict

# OpenRouter API configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "X-Title": "K-1 Tax Form Extraction"
}

# PDF processing settings
MAX_PDF_SIZE_MB = 20
MAX_RETRIES = 3

# Model settings
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
MAX_TOKENS = 16000
TEMPERATURE = 0.1

# File paths
PDFS_DIR = "pdfs"
EVAL_SET_FILE = "eval_set.csv"
MARKDOWN_OUTPUT_DIR = "markdown_output"
