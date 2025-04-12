from transformers import AutoModelForTokenClassification
from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT

def initialize_model():
    """
    Initialize a model for token classification.

    Returns:
        A model for token classification.
    """
    # 加载预训练的模型并指定标签映射
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,  # 模型检查点路径
        num_labels=len(ID_TO_LABEL),  # 标签数量
        id2label=ID_TO_LABEL,  # ID 到标签的映射
        label2id=LABEL_TO_ID,  # 标签到 ID 的映射
        ignore_mismatched_sizes=True,  # 忽略不匹配的大小
    )
    return model
