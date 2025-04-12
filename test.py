from dataset import build_dataset,preprocess_data
from utils import not_change_test_dataset
from tokenizer import initialize_tokenizer
import os
os.environ["http_proxy"] = 'http://127.0.0.1:8080'
os.environ['https_proxy'] = 'http://127.0.0.1:8080'

if __name__ == "__main__":
    # Test dataset
    raw_datasets = build_dataset()
    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"
    print("==" * 20)
    print(raw_datasets)
    print("--" * 20)
    print(raw_datasets['train'])
    print("--" * 20)
    print(raw_datasets['train'][0])
    tokenizer = initialize_tokenizer()
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)
    
    print("==" * 20)
    print(tokenized_datasets['train'])
    print("--" * 20)
    print(tokenized_datasets['train'][0])
    
    