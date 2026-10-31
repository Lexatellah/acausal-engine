import os
import sys

# Add the project root to sys.path so pytest can import acausal_engine
# without requiring an editable install.
sys.path.insert(0, os.path.dirname(__file__))
