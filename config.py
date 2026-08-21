"""
config.py

Loads environment variables and provides
project-wide configuration.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Read the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the key exists
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )