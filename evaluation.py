import evaluate
from transformers.trainer_utils import EvalPrediction

metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    """
    Compute evaluation metrics (precision, recall, f1) for predictions.

    Args:
        eval_predictions: Evaluation predictions.

    Returns:
        Dictionary with evaluation metrics. Keys: precision, recall, f1.
    """
    # 获取预测值和标签
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids

    # 对预测值取 argmax，得到每个 token 的预测标签
    predictions = predictions.argmax(axis=-1)

    # 将预测值和标签转换为字符串形式
    true_labels = [
        [metric_evaluator.id2label[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    pred_labels = [
        [metric_evaluator.id2label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    # 计算指标
    results = metric_evaluator.compute(predictions=pred_labels, references=true_labels)

    # 返回总体的 precision, recall 和 f1
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }