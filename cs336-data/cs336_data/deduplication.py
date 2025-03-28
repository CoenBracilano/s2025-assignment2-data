import hashlib
import os
from collections import defaultdict
from pathlib import Path
import random
import re
import shutil
from typing import List
import unicodedata

# Function to hash the line to a fixed-size key
def hash_line(line: str) -> str:
    return hashlib.sha256(line.encode('utf-8')).hexdigest()

def exact_line_dedup(input_files: list[os.PathLike], output_dir: os.PathLike)->list[os.PathLike]:
    # Initialize our hashmap to track the frequency of each line
    line_counter = defaultdict(int)
    # First pass, count occurences of each line
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_hash = hash_line(line)
                line_counter[line_hash] += 1
    
    os.makedirs(output_dir, exist_ok=True) # Check the output directory exists
    output_files = []

    #Second pass, process the input files and write unique lines to the output directory
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name)


        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        #Write unique lines 
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                line_hash = hash_line(line)
                if line_counter[line_hash] == 1:
                    f.write(line)
        # Add the path of the deduplicated file to the output list
        output_files.append(output_file_path)
    return output_files




# Since we dont know what we are going to recive as our input this function gives us a more standard set of text to work with
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # normalize whitespace
    return text.strip()

# Creates ngrams for us to use as an input for our minhash function 
def get_ngrams(text: str, n: int)-> set[str]:
    words = text.split()
    ngrams = set()
    for i in range(len(words) - 1):
        ngram = " ".join(words[i:i + n])
        ngrams.add(ngram)
    return ngrams   

# A helper function to compute the md5 hash of a given string using utf-8
def hash_with_seed(text: str, seed: int)-> int:
    return int(hashlib.md5(f"{seed}-{text}".encode("utf-8")).hexdigest(), 16)

# Compute the minhash signatures given a set of ngrams
def compute_signatures(ngrams: set[str], num_hashes: int)-> list[int]:
    signature = []

    # Itterate through the number of hashes provided
    for i in range(num_hashes):
        min_hash = float("inf")
        for ngram in ngrams: # Find the minimum output from our hash function accross the ngrams
            h = hash_with_seed(ngram, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature

# Given signatures and a number of bands, bucket the signatures if they are possible candiates for duplication
def lsh_buckets(signatures: dict[str, list[int]], num_bands: int) -> dict[str, set[str]]:
    band_size = len(next(iter(signatures.values()))) // num_bands
    buckets = defaultdict(set)
    candidates = defaultdict(set)

    for doc_id, signature in signatures.items():
        for band in range(num_bands):
            start = band * band_size
            end = start + band_size
            band_tuple = tuple(signature[start:end])
            band_hash = hash((band, band_tuple))  # tuple ensures different bands don’t collide
            buckets[band_hash].add(doc_id)

    # Collect candidate pairs from each bucket
    for bucket_docs in buckets.values():
        for doc1 in bucket_docs:
            for doc2 in bucket_docs:
                if doc1 != doc2:
                    candidates[doc1].add(doc2)

    return candidates

# Compute the jaccard simalarity of two sets 
def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0

# Filter out candidates based on their jaccard simalirty compared to a given threshold, output the clusters
def filter_candidates(ngram_sets: dict[str, set[str]], candidates: dict[str, set[str]], threshold: float) -> list[set[str]]:
    visited = set()
    clusters = []
    
    for doc1, doc_candidates in candidates.items():
        if doc1 in visited:
            continue

        cluster = {doc1}
        for doc2 in doc_candidates:
            if doc2 in visited:
                continue
            sim = jaccard_similarity(ngram_sets[doc1], ngram_sets[doc2])
            if sim >= threshold:
                cluster.add(doc2)

        if len(cluster) > 1:
            clusters.append(cluster)
            visited.update(cluster)

    return clusters

# Randomly remove all but one document from each cluster of duplicates that were above the threshold
def keep_docs(all_doc_ids: set[str], duplicate_clusters: list[set[str]]) -> set[str]:
    kept = set()
    excluded = set()
    for cluster in duplicate_clusters:
        retained = random.choice(list(cluster))
        kept.add(retained)
        excluded.update(cluster - {retained})
    # Add all docs that were not in any cluster
    not_duplicated = all_doc_ids - kept - excluded
    kept.update(not_duplicated)

    return kept
# Write the kept documents to the output directory using a set of id's for each document 
def write_deduplicated_files(input_paths: list[os.PathLike], kept_docs: set[str], output_directory: os.PathLike):
    os.makedirs(output_directory, exist_ok=True)

    for path in input_paths:
        doc_id = str(path)
        if doc_id in kept_docs:
            filename = os.path.basename(path)
            out_path = os.path.join(output_directory, filename)
            shutil.copyfile(path, out_path)

# Do fuzzy document deduplcation using minhash and LSH 
def minhash_dedup(input_files: list[os.PathLike], num_hashes: int, 
                  num_bands: int, ngrams: int, jaccard_threshold: float, output_directory: os.PathLike)-> list[os.PathLike]:
    # Open the files and normalize the text
    # Then compute their ngrams
    doc_texts = {}
    ngram_sets = {}
    for path in input_files:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        norm = normalize_text(raw)
        ngram_set = get_ngrams(norm, ngrams)
        doc_id = str(path)
        doc_texts[doc_id] = raw
        ngram_sets[doc_id] = ngram_set

    # Using the ngrams generated compute MinHash signatures
    signatures = {
        doc_id: compute_signatures(ngrams, num_hashes)
        for doc_id, ngrams in ngram_sets.items()
    }

    #LSH to find candidate duplicates
    candidates = lsh_buckets(signatures, num_bands)
    #Filter documents by jaccard similarity using the given threshold
    clusters = filter_candidates(ngram_sets, candidates, jaccard_threshold)
    # Keep one document per cluster
    kept_docs = keep_docs(set(doc_texts.keys()), clusters)

    #Write deduplicated documents to output
    os.makedirs(output_directory, exist_ok=True)
    output_paths = []

    for path in input_files:
        doc_id = str(path)
        if doc_id in kept_docs:
            filename = os.path.basename(path)
            out_path = os.path.join(output_directory, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(doc_texts[doc_id])
            output_paths.append(Path(out_path))

    return output_paths






if __name__ == "__main__":
 x=0