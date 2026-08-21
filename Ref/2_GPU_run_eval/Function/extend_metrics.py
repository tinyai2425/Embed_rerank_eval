import re
from difflib import SequenceMatcher
from collections import defaultdict
import math
from transformers import AutoTokenizer

# 延迟初始化
TOKENIZER_CONFIG_PATH = None
tokenizer = None

def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)

def GET_answer(answer):
    non_empty_splits = [split for split in answer.split("\n") if split.strip() != ""]
    if len(non_empty_splits) > 0:
        return bool(re.search("答案", non_empty_splits[-1]))
    else:
        return False

def split_sentences(text: str) -> list:
    sentence_delimiters = r'[。！？；;.!?\n\r]+'
    sentences = [
        s.strip() 
        for s in re.split(sentence_delimiters, text) 
        if s.strip() and len(s.strip()) >= 1
    ]
    return sentences

def calculate_repetition_rate(text: str, similarity_threshold: float = 0.85) -> float:
    sentences = split_sentences(text)
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return 0.0
    
    groups = defaultdict(list)
    for sent in sentences:
        matched = False
        for key in groups:
            if SequenceMatcher(None, key, sent).ratio() >= similarity_threshold:
                groups[key].append(sent)
                matched = True
                break
        if not matched:
            groups[sent].append(sent)
    
    repeated_chars = sum(
        sum(len(sent) for sent in group[1:])
        for group in groups.values() 
        if len(group) > 1
    )
    return repeated_chars / total_chars

def calculate_token_entropy(text: str) -> float:
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Please call init_tokenizer(path) before using calculate_token_entropy.")

    vocab_size = tokenizer.vocab_size
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0
    
    counts = defaultdict(int)
    entropy_sum = 0.0
    
    for i, token in enumerate(tokens):
        count = counts[token]
        probability = (count + 1) / (i + vocab_size)
        entropy_sum += -math.log2(probability)
        counts[token] += 1
    
    return entropy_sum / total_tokens
