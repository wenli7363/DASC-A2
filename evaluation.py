import evaluate
from transformers.trainer_utils import EvalPrediction
from constants import ID_TO_LABEL  # 确保从 constants.py 导入 ID_TO_LABEL

from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids
    predictions = predictions.argmax(axis=-1)

    true_labels = [
        [ID_TO_LABEL[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    pred_labels = [
        [ID_TO_LABEL[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "f1": f1_score(true_labels, pred_labels, zero_division=0),
    }