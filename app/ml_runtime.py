"""Compatibility alias for the app runtime ML facade."""

from __future__ import annotations

import sys

from app.runtime import ml as _runtime_ml

sys.modules[__name__] = _runtime_ml
