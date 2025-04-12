import evaluate
import os
os.environ["http_proxy"] = 'http://127.0.0.1:8080'
os.environ['https_proxy'] = 'http://127.0.0.1:8080'
try:
    metric = evaluate.load("seqeval")
    print("Seqeval loaded successfully!")
except Exception as e:
    print(f"Error loading seqeval: {e}")