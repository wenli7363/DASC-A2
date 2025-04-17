from transformers import Trainer, TrainingArguments
from dataset import get_data_collator
from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        dataloader_num_workers=4,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        eval_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        learning_rate=1e-5 * 2,
        save_strategy="epoch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=True,
        eval_accumulation_steps=1000,
    )

    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for token classification.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.
    """
    data_collator = get_data_collator(tokenizer)

    training_args: TrainingArguments = create_training_arguments()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
