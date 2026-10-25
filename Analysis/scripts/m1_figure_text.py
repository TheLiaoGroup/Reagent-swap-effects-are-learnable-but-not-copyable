"""Display-only typography shared by plot and SVG figure labels."""
from __future__ import annotations
import re

def numeric_minus(text):
    """Use a minus for signed numbers while preserving name and range hyphens."""
    return re.sub('(?<![\\w-])-(?=\\d|\\.\\d)', '−', text)
