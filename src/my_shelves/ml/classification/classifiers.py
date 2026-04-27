#pylint: disable=too-many-arguments
#pylint: disable=too-many-positional-arguments

"""
NLP classification module providing reusable pipeline wrappers for sentiment analysis,
emotion detection, and zero-shot classification using HuggingFace Transformers.

All classifiers share a common base class (BaseClassifier) that handles input validation,
batch inference, and optional CSV persistence. Subclasses implement task-specific
postprocessing logic.

Classes:
    BaseClassifier: Abstract base class for all HuggingFace pipeline classifiers.
    SentimentClassifier: Predicts positive/negative/neutral sentiment.
    EmotionClassifier: Detects one or more emotions with a confidence threshold.
    ZeroShotClassifier: Classifies text against arbitrary candidate labels without fine-tuning.
"""
import os
import time
from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
from my_shelves.ml.classification.dict_features import dict_labels
from my_shelves.utils.csv_creator import append_unique_rows



class BaseClassifier:
    """
    Abstract base class for HuggingFace pipeline-based text classifiers.

    Handles pipeline initialization, input validation, batched inference,
    and optional persistence to CSV. Subclasses must implement `_postprocess`.

    Args:
        task (str): HuggingFace pipeline task name (e.g. 'sentiment-analysis').
        model_path (str): Path or HuggingFace Hub identifier for the model and tokenizer.
        top_k (int | None): Number of top predictions to return per input. None returns all.
        batch_size (int): Number of samples processed per inference batch.
        device (int): Device index for inference. -1 for CPU, 0+ for GPU.
    """

    def __init__(self,
                 task: str,
                 model_path: str,
                 top_k: int | None = None,
                 batch_size: int = 16,
                 device: int = -1):

        self.task = task
        self.model_path = model_path
        self.top_k = top_k
        self.batch_size = batch_size
        self.device = device

        self.classifier = pipeline(
            task,
            model=model_path,
            tokenizer=model_path,
            top_k=top_k,
            truncation=True,
            max_length=512,
            device=device
        )

    def _validate_inputs(self, text: pd.Series, ids: pd.Series):
        """
        Validate and normalize input Series before inference.

        Ensures that `text` and `ids` have the same length, fills NaN values
        in `text` with empty strings, and extracts column names for downstream use.

        Args:
            text (pd.Series): Series of raw input texts.
            ids (pd.Series): Series of unique identifiers aligned with `text`.

        Returns:
            tuple[pd.Series, pd.Series, str, str]:
                Cleaned text Series, ids Series, id column name, text column name.

        Raises:
            ValueError: If `text` and `ids` do not have the same length.
        """
        if len(text) != len(ids):
            raise ValueError("text et ids doivent avoir la même longueur")

        text = text.fillna("").astype(str)

        id_col = ids.name or "id"

        return text, ids, id_col

    def _predict_batches(self, texts):
        """
        Run batched inference over a list of texts using the HuggingFace pipeline.

        Iterates over `texts` in chunks of `self.batch_size`, displaying a tqdm
        progress bar. Results from all batches are concatenated and returned.

        Args:
            texts (list[str]): List of input strings to classify.

        Returns:
            list: Raw prediction outputs from the pipeline, one entry per input text.
        """
        results = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Inference"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.classifier(batch)
            results.extend(batch_results)

        return results

    def _postprocess(self, results, ids, id_col):
        """
        Convert raw pipeline outputs into a structured DataFrame.

        Must be implemented by each subclass to apply task-specific logic
        (label extraction, thresholding, multi-label handling, etc.).

        Args:
            results (list): Raw outputs returned by `_predict_batches`.
            ids (pd.Series): Original identifier Series aligned with results.
            id_col (str): Column name to use for the identifier in the output DataFrame.

        Returns:
            pd.DataFrame: Processed predictions with at least one id column and
                          one or more prediction columns.

        Raises:
            NotImplementedError: Always, until overridden by a subclass.
        """
        raise NotImplementedError

    def predict(self, text: pd.Series, ids: pd.Series) -> pd.DataFrame:
        """
        Validate inputs, run batched inference, postprocess results, and return a DataFrame.

        This is the main entry point for inference. Execution time is measured and
        printed to stdout upon completion.

        Args:
            text (pd.Series): Series of input texts to classify.
            ids (pd.Series): Series of unique identifiers aligned with `text`.

        Returns:
            pd.DataFrame: Postprocessed predictions as returned by `_postprocess`.
        """
        start = time.time()

        text, ids, id_col = self._validate_inputs(text, ids)

        results = self._predict_batches(text.tolist())

        df = self._postprocess(results, ids, id_col)

        duration = round(time.time() - start, 2)
        print(f"Inference terminée en {duration}s ({len(text)} lignes)")

        return df

    def predict_and_save(self,
                     text: pd.Series,
                     ids: pd.Series,
                     filename: str,
                     base_dir: str = None) -> pd.DataFrame:

      # Charger les IDs déjà traités
      if os.path.isfile(filename):
          existing_ids = pd.read_csv(filename, usecols=["book_id"])["book_id"]
          mask = ~ids.isin(existing_ids)
          text = text[mask]
          ids = ids[mask]

      if ids.empty:
          print("Aucun nouveau book_id à traiter.")
          return pd.DataFrame()

      df = self.predict(text, ids)
      append_unique_rows(df, filename)
      return df


class SentimentClassifier(BaseClassifier):
    """
    Classifier for positive / negative / neutral sentiment analysis.

    Uses a RoBERTa model fine-tuned on Twitter data by default. Inherits batched
    inference from BaseClassifier and extracts the single highest-scoring label
    per input.

    Args:
        model_path (str): HuggingFace model identifier. Defaults to
            'cardiffnlp/twitter-roberta-base-sentiment-latest'.
        top_k (int | None): Number of top predictions to return. None returns all.
        batch_size (int): Number of samples per inference batch. Defaults to 32.
        device (int): Device index. -1 for CPU, 0+ for GPU. Defaults to -1.
    """

    def __init__(self,
                 model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                 top_k=None,
                 batch_size=32,
                 device=-1):

        super().__init__(
            task="sentiment-analysis",
            model_path=model_path,
            top_k=top_k,
            batch_size=batch_size,
            device=device
        )

    def _postprocess(self, results, ids, id_col):
        """
        Extract the top sentiment label and its confidence score for each input.

        Args:
            results (list): Raw pipeline outputs, each being a dict or a list of dicts
                            with 'label' and 'score' keys.
            ids (pd.Series): Identifier Series aligned with results.
            id_col (str): Column name for the identifier in the output DataFrame.

        Returns:
            pd.DataFrame: One row per input with columns:
                - <id_col>: original identifier.
                - sentiment (str): predicted sentiment label.
                - sentiment_score (float): confidence score for the predicted label.
        """
        rows = []

        for i, res in enumerate(results):

            best = res[0] if isinstance(res, list) else res

            rows.append({
                id_col: ids.iloc[i],
                "sentiment": best["label"],
                "sentiment_score": best["score"]
            })

        return pd.DataFrame(rows)


class EmotionClassifier(BaseClassifier):
    """
    Multi-label emotion classifier with configurable confidence thresholding.

    Predicts one or more emotions per input text. Labels whose score meets or
    exceeds `threshold` are included; if none qualify, the top prediction is
    used as a fallback.

    Args:
        model_path (str): HuggingFace model identifier. Defaults to
            'j-hartmann/emotion-english-distilroberta-base'.
        threshold (float): Minimum confidence score to include a label. Defaults to 0.4.
        top_k (int | None): Number of top predictions to return from the pipeline.
            None returns all available labels.
        batch_size (int): Number of samples per inference batch. Defaults to 32.
        device (int): Device index. -1 for CPU, 0+ for GPU. Defaults to -1.
    """

    def __init__(self,
                 model_path: str = "j-hartmann/emotion-english-distilroberta-base",
                 threshold: float = 0.4,
                 top_k: int | None = None,
                 batch_size: int = 32,
                 device: int = -1):

        super().__init__(
            task="sentiment-analysis",
            model_path=model_path,
            top_k=top_k,
            batch_size=batch_size,
            device=device
        )

        self.threshold = threshold

    def _postprocess(self, results, ids, id_col):
        """
        Apply threshold filtering to select active emotion labels for each input.

        Labels with a score >= `self.threshold` are collected. If no label meets
        the threshold, the highest-scoring label is used as a fallback.

        Args:
            results (list): Raw pipeline outputs, each being a list of dicts
                            with 'label' and 'score' keys, sorted by score descending.
            ids (pd.Series): Identifier Series aligned with results.
            id_col (str): Column name for the identifier in the output DataFrame.

        Returns:
            pd.DataFrame: One row per input with columns:
                - <id_col>: original identifier.
                - emotions (list[str]): labels that met or exceeded the threshold
                  (or the fallback top label).
                - top_emotion (str | None): highest-scoring label.
                - top_score (float | None): score of the highest-scoring label.
        """
        rows = []

        for i, emotion_list in enumerate(results):

            if isinstance(emotion_list, dict):
                emotion_list = [emotion_list]

            selected = [
                e["label"]
                for e in emotion_list
                if e["score"] >= self.threshold
            ]

            if not selected and emotion_list:
                selected = [emotion_list[0]["label"]]

            rows.append({
                id_col: ids.iloc[i],
                "emotions": selected,
                "top_emotion": emotion_list[0]["label"] if emotion_list else None,
                "top_score": emotion_list[0]["score"] if emotion_list else None
            })

        return pd.DataFrame(rows)


class ZeroShotClassifier(BaseClassifier):
    """
    Zero-shot text classifier using a Natural Language Inference (NLI) model.

    Labels, hypothesis template, and multi-label mode are loaded from `dict_labels`
    using the provided task key. No fine-tuning is required: the model scores each
    candidate label against the input text via entailment.

    Dual thresholds control output confidence:
    - In multi-label mode, all labels above `threshold_high` are returned; if none
      qualify, the best label is included if it meets `threshold_low`.
    - In single-label mode, the best label is returned only if it meets `threshold_low`.

    Args:
        task (str): Key used to look up label configuration in `dict_labels`.
        model_path (str): HuggingFace model identifier. Defaults to
            'valhalla/distilbart-mnli-12-3'.
        threshold_high (float): High-confidence cutoff for multi-label selection.
            Defaults to 0.80.
        threshold_low (float): Minimum score to include the best label as a fallback.
            Defaults to 0.60.
        batch_size (int): Number of samples per inference batch. Defaults to 16.
        device (int): Device index. -1 for CPU, 0+ for GPU. Defaults to -1.
    """

    def __init__(self,
                 task: str,
                 model_path: str = "valhalla/distilbart-mnli-12-3",
                 threshold_high: float = 0.80,
                 threshold_low: float = 0.60,
                 batch_size: int = 16,
                 device: int = -1):

        config = dict_labels[task]

        super().__init__(
            task="zero-shot-classification",
            model_path=model_path,
            top_k=None,
            batch_size=batch_size,
            device=device
        )

        self.labels = config["labels"]
        self.multi_label = config["multi_label"]
        self.hypothesis_template = config["hypothesis_template"]

        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def predict(self, text: pd.Series, ids: pd.Series) -> pd.DataFrame:
        """
        Run zero-shot classification with task-specific candidate labels and template.

        Overrides `BaseClassifier.predict` to pass `candidate_labels`,
        `hypothesis_template`, and `multi_label` directly to the HuggingFace pipeline,
        which is required for zero-shot classification.

        Args:
            text (pd.Series): Series of input texts to classify.
            ids (pd.Series): Series of unique identifiers aligned with `text`.

        Returns:
            pd.DataFrame: Postprocessed predictions as returned by `_postprocess`.
        """
        start = pd.Timestamp.now()

        text, ids, id_col = self._validate_inputs(text, ids)

        results = []

        for i in range(0, len(text), self.batch_size):
            batch = text.iloc[i:i + self.batch_size].tolist()

            batch_results = self.classifier(
                batch,
                candidate_labels=self.labels,
                hypothesis_template=self.hypothesis_template,
                multi_label=self.multi_label
            )

            results.extend(batch_results)

        df = self._postprocess(results, ids, id_col)

        duration = (pd.Timestamp.now() - start).total_seconds()
        print(f"Zero-shot inference en {round(duration, 2)}s")

        return df

    def _postprocess(self, results, ids, id_col):
        """
        Apply dual-threshold filtering to select final labels for each input.

        In multi-label mode, all labels scoring >= `threshold_high` are selected.
        If none qualify, the best label is included if it scores >= `threshold_low`.
        In single-label mode, the best label is returned only if it meets `threshold_low`.
        An empty list is returned when no label clears the threshold.

        Args:
            results (list): Raw zero-shot pipeline outputs. Each entry is a dict with
                            'labels' (list[str]) and 'scores' (list[float]) sorted
                            by score descending.
            ids (pd.Series): Identifier Series aligned with results.
            id_col (str): Column name for the identifier in the output DataFrame.

        Returns:
            pd.DataFrame: One row per input with columns:
                - <id_col>: original identifier.
                - labels (list[str]): selected labels after threshold filtering.
                - best_label (str): highest-scoring candidate label.
                - best_score (float): confidence score for the best label.
        """
        rows = []

        for i, res in enumerate(results):

            labels_list = res["labels"]
            scores_list = res["scores"]

            best_label = labels_list[0]
            best_score = scores_list[0]

            if self.multi_label:
                high_conf = [
                    l for l, s in zip(labels_list, scores_list)
                    if s >= self.threshold_high
                ]
                final = high_conf or (
                    [best_label] if best_score >= self.threshold_low else []
                )
            else:
                final = [best_label] if best_score >= self.threshold_low else []

            rows.append({
                id_col: ids.iloc[i],
                "labels": final,
                "best_label": best_label,
                "best_score": best_score
            })

        return pd.DataFrame(rows)
