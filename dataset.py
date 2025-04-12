from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForTokenClassification

def align_labels_with_tokens(
    labels: list[int],
    word_ids: list[int | None],
) -> list[int]:
    """
    Align labels with tokenized word IDs, ensuring special tokens are ignored (-100).

    Args:
        labels: List of token labels.
        word_ids: List of word IDs.

    Returns:
        A list of aligned labels.
    """
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:  # Special tokens
            aligned_labels.append(-100)
        elif word_id != previous_word_id:  # Start of a new word
            aligned_labels.append(labels[word_id])
        else:  # Inside a word
            # Inherit the label from the first token of the word
            aligned_labels.append(aligned_labels[-1])  # Use the last appended label
        previous_word_id = word_id

    return aligned_labels


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    Tokenize input examples and align labels for token classification.

    To preprocess our whole dataset, we need to tokenize all the inputs and apply align_labels_with_tokens() on all the labels.
    To take advantage of the speed of our fast tokenizer, it’s best to tokenize lots of texts at the same time,
    so this function will processes a list of examples and return a list of tokenized inputs with aligned labels.

    Args:
        examples: Input examples.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    aligned_labels = [
        align_labels_with_tokens(
            labels=labels,
            word_ids=tokenized_inputs.word_ids(i),
        )
        for i, labels in enumerate(all_labels)
    ]
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.


    Below is an example of how to load a dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')
    ```

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is the same as the test split from the MultiCoNER NER dataset,
    Which means that:
        raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    """
    # Write your code here.
    print("Start Loading dataset...")
    raw_datasets: DatasetDict = DatasetDict()
    raw_datasets["train"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="train")
    raw_datasets["validation"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="validation")
    raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    return raw_datasets


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets



def get_data_collator(tokenizer) -> DataCollatorForTokenClassification:
    """
    Get the data collator for token classification.

    Returns:
        Data collator.
    """
    # Write your code here.
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length = None,
        return_tensors="pt",
    )
    return data_collator