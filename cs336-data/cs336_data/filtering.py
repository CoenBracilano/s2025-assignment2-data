import unicodedata
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from typing import Any
import fasttext
import re
import nltk
import math

# Use resiliparse functions to detect the encoding and then extract the raw text from given byte stream
def extract_text(input: bytes) -> str | None:
    enc = detect_encoding(input)
    uni_str = input.decode(enc, errors="replace")  # Decode safely
    out = extract_plain_text(uni_str)

    # Normalize Unicode characters
    out = unicodedata.normalize("NFC", out)

    #Remove trailing spaces from lines
    out = "\n".join(line.rstrip() for line in out.splitlines())

    # Collapse excessive newlines (optional for safety)
    out = re.sub(r'\n{3,}', '\n\n', out)

    #Strip trailing whitespace and enforce exactly 2 newlines at the end
    out = out.rstrip()

    return out



def language_id(text: str)-> tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\required_docs\lid.176.bin"
    model = fasttext.load_model(model_path) #Load the downloaded model
    clean_text = text.replace("\n", " ") # Fasttext cant handle newlines so we remove them 
    prediction = model.predict(clean_text, k=1) 
    # Guess the language, k=1 asks for the top language, where k is the number of languages you want the model to guess

    language = prediction[0][0].replace("__label__", "") # Reformat the automatic output and get what we want 
    confidence = prediction[1][0]
    return language, confidence # Return the values

# Apply the regex function to the file to mask emails
def mask_email(text: str)-> tuple[str, int]:
    exp = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    replacement = "|||EMAIL_ADDRESS|||"
    out = re.subn(exp, replacement, text)
    return out

# Apply the regex function to the file to mask phone numbers
def mask_phone(text: str)-> tuple[str, int]:
    exp = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    replacment = "|||PHONE_NUMBER|||"
    out = re.subn(exp, replacment, text)
    return out
# Apply the regex function to the file to mask ipv4 addresses
def mask_ipv4(text: str)-> tuple[str, int]:
    exp = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    replacment = "|||IP_ADDRESS|||"
    out = re.subn(exp, replacment, text)
    return out
# Combine all the PII masking functions and apply them to a given string of text
def mask_PII(text: str)-> str:
    temp = mask_email(text)
    text = temp[0]
    temp = mask_ipv4(text)
    text = temp[0]
    temp = mask_phone(text)
    return temp[0]


def find_NSFW(text: str)->tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\required_docs\jigsaw_fasttext_bigrams_nsfw_final.bin"
    model = fasttext.load_model(model_path)
    clean_text = text.replace("\n", " ") # Fasttext doesnt like newlines 
    prediction = model.predict(clean_text, k=1) # Predict if the text is NSFW

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return label, confidence


def find_toxic(text: str)->tuple[Any, float]:
    model_path = r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\required_docs\jigsaw_fasttext_bigrams_hatespeech_final.bin"
    model = fasttext.load_model(model_path)
    clean_text = text.replace("\n", " ") # Fasttext doesnt like newlines 
    prediction = model.predict(clean_text, k=1) # Predict if the text is "toxic"

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]

    return label, confidence 
    
# Apply the 4 rules given by the "Gopher paper" 
def gopher_filter(text: str)-> bool:
    tokens = nltk.word_tokenize(text)
    word_len = 0
    # First gopher rule, number of words
    if len(tokens) < 50 or len(tokens) > 100000:
        print("Number of words error")
        return False
    
    word_len = sum(len(word) for word in tokens)
    
    mean_word_len = math.floor((word_len/len(tokens)))
    if mean_word_len < 3 or mean_word_len > 10:
        print("Word length Error")
        return False
    # Second gopher rule, mean word length between 3 and 10 

    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.30:
            print("Elipse Error")
            return False
    # Third gopher rule, less than 30% of lines can end with an elipse

    alpha_words = sum(1 for word in tokens if re.search(r'[A-Za-z]', word))
    if alpha_words / len(tokens) < 0.80:
        print("alphabetic error")
        return False
    # Fourth gopher rule more than 80% of words must contain at least 1 alphabetic character
    return True
    
# Train a fasttext quality classifier on the given input file and parameters
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


# Using the trained text classifier, label the given string with wiki if it is high quality or cc if it is low quality
# return the label and the confidence value
def label_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\required_docs\quality_classifier.ftz")
    cleaned_text = text.replace("\n", " ").strip()

    labels, probabilities = model.predict(cleaned_text, k=1)
    raw_label = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    label = "wiki" if raw_label == "high" else "cc"
    return (label, confidence)



