"""
Configuration module for zero-shot classification tasks and candidate NLI models.

`dict_labels` defines the label space, hypothesis template, and multi-label mode
for each classification task supported by ZeroShotClassifier.

`models` lists HuggingFace NLI model identifiers that can be passed as `model_path`
to ZeroShotClassifier, ordered roughly from most accurate to fastest.
"""

# Registry of zero-shot classification task configurations.
#
# Each key is a task name (str) used as the `task` argument in ZeroShotClassifier.
# Each value is a dict with the following keys:
#
#     labels (list[str]):
#         Candidate labels passed to the zero-shot pipeline. Phrased to complete
#         the hypothesis template naturally.
#
#     hypothesis_template (str):
#         Template string containing a single '{}' placeholder. The pipeline
#         fills it with each candidate label to form an NLI hypothesis
#         (e.g. "The topic of this book is about love and romantic relationships.").
#
#     multi_label (bool):
#         If True, multiple labels can be active simultaneously (scores are
#         computed independently via sigmoid). If False, labels are mutually
#         exclusive (scores are normalized via softmax).
#
# Tasks
# -----
# content_intensity : multi-label
#     Flags disturbing or sensitive content categories present in the book.
# romance_heat_level : single-label
#     Rates the explicitness of romantic content on a four-point scale.
# character_type : single-label
#     Characterizes the moral profile of the main protagonist(s).
# main_themes : multi-label
#     Identifies the primary thematic concerns of the book.
# rhythm : single-label
#     Describes the narrative pacing style of the book.
dict_labels = {
    'content_intensity': {
        'labels': [
            "no disturbing or sensitive content",
            "scenes involving physical violence or injury",
            "scenes involving death or dying",
            "scenes involving sexual violence or assault",
            "scenes involving self-harm or suicide",
            "scenes involving psychological abuse or distress",
            "scenes involving war, torture, or extreme cruelty"
        ],
        'hypothesis_template': "The content intensity of this book is {}.",
        'multi_label': True
    },

    'romance_heat_level': {
        'labels': [
            "no romance at all",
            "clean and innocent romance with no explicit scenes",
            "romance with some mild intimacy or suggestive scenes",
            "explicit sexual scenes described in detail"
        ],
        'hypothesis_template': "This romance in this book contains {}.",
        'multi_label': False
    },

    'character_type': {
        'labels': [
            "a traditional heroic main character",
            "an anti-hero with morally ambiguous behavior",
            "a story with multiple main characters",
            "a main character who is a villain"
        ],
        'hypothesis_template': "This story features {}.",
        'multi_label': False
    },

    'main_themes': {
        'labels': [
            "love and romantic relationships",
            "war and conflict",
            "travel and exploration",
            "a personal identity quest",
            "social justice issues",
            "ecology and environmental concerns",
            "technology and its impact",
            "death and loss",
            "mystery and investigation"
        ],
        'hypothesis_template': "The topic of this book is about {}.",
        'multi_label': True
    },

    'rhythm': {
        'labels': [
            "a fast-paced story with constant action",
            "a slow-paced story focused on atmosphere",
            "an uneven pacing with ups and downs",
            "a balanced pacing throughout the story"
        ],
        'hypothesis_template': "The narrative pacing of this book is {}.",
        'multi_label': False
    }
}

# HuggingFace NLI model identifiers compatible with ZeroShotClassifier.
# Ordered roughly from most accurate to fastest.
models = [
    'facebook/bart-large-mnli',               # Most accurate, slowest
    'cross-encoder/nli-deberta-v3-large',     # High accuracy, slower
    'typeform/distilbert-base-uncased-mnli',  # Lightweight, faster
    'valhalla/distilbart-mnli-12-3'           # Good speed/accuracy trade-off (default)
]
