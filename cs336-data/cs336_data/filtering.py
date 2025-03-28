import random
import unicodedata
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from typing import Any
import gzip
import chardet
import fasttext
import re
import nltk
import math
from warcio.archiveiterator import ArchiveIterator


def extract_text(input: bytes) -> str | None:
    enc = detect_encoding(input)
    print(f"[DEBUG] Detected encoding: {enc}")
    uni_str = input.decode(enc, errors="replace")  # Decode safely
    out = extract_plain_text(uni_str)

    # Normalize Unicode characters
    out = unicodedata.normalize("NFC", out)

    # Remove trailing spaces from lines
    out = "\n".join(line.rstrip() for line in out.splitlines())

    # Collapse excessive newlines (optional for safety)
    out = re.sub(r'\n{3,}', '\n\n', out)

    # Strip trailing whitespace and enforce exactly 2 newlines at the end
    out = out.rstrip()

    return out



def language_id(text: str)-> tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\lid.176.bin"
    model = fasttext.load_model(model_path) #Load the downloaded model
    clean_text = text.replace("\n", " ") # Fasttext cant handle newlines so we remove them 
    prediction = model.predict(clean_text, k=1) 
    # Guess the language, k=1 asks for the top language, where k is the number of languages you want the model to guess

    language = prediction[0][0].replace("__label__", "") # Reformat the automatic output and get what we want 
    confidence = prediction[1][0]
    return language, confidence # Return the values


def mask_email(text: str)-> tuple[str, int]:
    exp = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    replacement = "|||EMAIL_ADDRESS|||"
    out = re.subn(exp, replacement, text)
    return out

def mask_phone(text: str)-> tuple[str, int]:
    exp = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    replacment = "|||PHONE_NUMBER|||"
    out = re.subn(exp, replacment, text)
    return out

def mask_ipv4(text: str)-> tuple[str, int]:
    exp = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    replacment = "|||IP_ADDRESS|||"
    out = re.subn(exp, replacment, text)
    return out

def mask_PII(text: str)-> str:
    temp = mask_email(text)
    text = temp[0]
    temp = mask_ipv4(text)
    text = temp[0]
    temp = mask_phone(text)
    return temp[0]


def find_NSFW(text: str)->tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\jigsaw_fasttext_bigrams_nsfw_final.bin"
    model = fasttext.load_model(model_path)
    clean_text = text.replace("\n", " ")
    prediction = model.predict(clean_text, k=1)

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return label, confidence


def find_toxic(text: str)->tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\jigsaw_fasttext_bigrams_hatespeech_final.bin"
    model = fasttext.load_model(model_path)
    clean_text = text.replace("\n", " ")
    prediction = model.predict(clean_text, k=1)

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return label, confidence 
    
def gopher_filter(text: str)-> bool:
    tokens = nltk.word_tokenize(text)
    word_len = 0
    # First gopher rule, number of words
    if len(tokens) < 50 or len(tokens) > 100000:
        return False
    
    word_len = sum(len(word) for word in tokens)
    
    mean_word_len = math.floor((word_len/len(tokens)))
    if mean_word_len < 3 or mean_word_len > 10:
        return False
    # Second gopher rule, mean word length between 3 and 10 

    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.30:
            return False
    # Third gopher rule, less than 30% of lines can end with an elipse

    alpha_words = sum(1 for word in tokens if re.search(r'[A-Za-z]', word))
    if alpha_words / len(tokens) < 0.80:
        return False
    # Fourth gopher rule more than 80% of words must contain at least 1 alphabetic character
    return True
    

def train_classifier(train_file=r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\balanced_train.txt",
    model_path=r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\quality_classifier.ftz",
    lr=0.5,
    epoch=10,
    wordNgrams=2,
    minn=3,
    maxn=6):
    print("Training fastText classifier...")
    model = fasttext.train_supervised(
        input=train_file,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        minn=minn,
        maxn=maxn,
        verbose=2
    )

    print(f"Training complete. Saving model to {model_path}")
    model.save_model(model_path)
    return model
    

def subsample_urls(input_path, output_path, num_urls=10000):
    with open(input_path, 'r', encoding='utf-8') as f:
        all_urls = f.readlines()

    print(f"Total URLs available: {len(all_urls)}")

    sampled = random.sample(all_urls, min(num_urls, len(all_urls)))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(sampled)

    print(f"Wrote {len(sampled)} sampled URLs to {output_path}")

def extract_docs(warc_path, output_path, max_docs=3000):
    count = 0
    processed = 0
    with gzip.open(warc_path, "rb") as stream, open(output_path, "w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue
                
            processed += 1
            # Raw response bytes (HTML and headers)
            try:
                raw_bytes = record.content_stream().read()
                text = extract_text(raw_bytes)
            except Exception as e:
                continue  # skip malformed or non-text responses

            # Run your checks
            language, score = language_id(text)
            if language != "en":
                print("Not in english")
                continue
            if score < 0.5:
                print("Language score too low")
                continue
            # if not gopher_filter(text):
            #     continue
            # label, confidence = find_NSFW(text)
            # if label == "nsfw" and confidence > 0.3:
            #     print("Doc is NSFW")
            #     continue
            # label, confidence = find_toxic(text)
            # if label == "toxic" and confidence > 0.3:
            #     print("Doc is toxic")
            #     continue


            # Remove PII from the document
            # text = mask_PII(text)

            # Format and write line
            text = text.replace("\n", " ").strip()
            if not text:
                continue
            out.write(f"__label__low {text}\n")

            count += 1
            if count % 100 == 0 or processed % 250 == 0:
                print(f"Processed {processed} docs, accepted {count}")

            if count >= max_docs:
                break

    print(f"Done, {count} examples written to {output_path}")


def label_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\quality_classifier.ftz")
    cleaned_text = text.replace("\n", " ").strip()

    labels, probabilities = model.predict(cleaned_text, k=1)
    raw_label = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    label = "wiki" if raw_label == "high" else "cc"
    return (label, confidence)

def temp():
    with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\negative_text.txt", "r", encoding="utf-8") as f:
        neg_lines = f.readlines()
    balanced_neg = random.sample(neg_lines, 1400)
    with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\balanced_train.txt", "w", encoding="utf-8") as out:
        with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\positive_text.txt", "r", encoding="utf-8") as f:
            out.writelines(f.readlines())
        out.writelines(balanced_neg)

if __name__ == "__main__":
    #extract_docs(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\SampleWARC.gz",
    #              r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\negative_text.txt")
    # subsample_urls(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\enwiki-20240420-extracted_urls.txt", 
    # r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\subsampled_positive_urls.txt")
    x=0
    # model = train_classifier()






