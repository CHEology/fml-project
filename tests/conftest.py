"""Shared pytest setup for native ML libraries.

Import torch before FAISS-backed tests run. On macOS, initializing FAISS first
can make later BatchNorm calls in torch abort inside native code.
"""

import torch  # noqa: F401
