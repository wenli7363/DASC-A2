import os
import logging

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds


def setup_logging(log_file="training.log"):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    """
    Main function to execute model training and evaluation.
    """
    # Set up logging
    setup_logging()

    logging.info("Starting the training process...")

    # Set random seeds for reproducibility
    set_random_seeds()
    logging.info("Random seeds set.")

    # Initialize tokenizer and model
    model = initialize_model()
    logging.info("Model initialized.")

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    logging.info("Tokenizer initialized.")

    raw_datasets = build_dataset()
    logging.info("Datasets loaded.")

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"
    logging.info("Test dataset integrity verified.")

    # Load and preprocess datasets
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)
    logging.info("Datasets preprocessed.")

    # Build and train the model
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    logging.info("Trainer initialized. Starting training...")
    trainer.train()
    logging.info("Training completed.")

    torch.cuda.empty_cache()
    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    logging.info(f"Test Metrics: {test_metrics}")
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
