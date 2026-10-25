"""Reader-facing analysis names; scientific inputs and provenance stay intact."""
from __future__ import annotations
import re
NAMES = {'REFERENCE_LEARNING': 'reference-learning', 'RECIPE_OPTIONS': 'recipe-graph', 'RESPONSE_LEARNING': 'exact-swap'}
SOURCE_COLUMNS = {'source', 'source_file', 'provenance_source_file'}

def reader_text(text):
    text = text.replace('RECIPE_OPTIONS strict records', 'recipe-graph eligible records')
    text = text.replace('RESPONSE_LEARNING exact-matching records', 'exact-swap matching records')
    text = re.sub('\\b(?:REFERENCE_LEARNING|RECIPE_OPTIONS|RESPONSE_LEARNING)\\b', lambda match: NAMES[match[0]], text)
    text = text.replace('covariance-screen', 'reference-learning')
    text = text.replace('recipe-graph strict records', 'recipe-graph eligible records')
    text = text.replace('exact-swap exact-matching records', 'exact-swap matching records')
    return text.replace('. reference-learning partition', '. Reference-learning partition')

def reader_table(data):
    shown = data.copy()
    for column in shown:
        if column in SOURCE_COLUMNS or 'path' in column:
            continue
        if shown[column].dtype == object:
            shown[column] = shown[column].map(lambda value: reader_text(value) if isinstance(value, str) else value)
    return shown
