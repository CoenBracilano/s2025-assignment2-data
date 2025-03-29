
import gzip
import random

from fastwarc import ArchiveIterator
from cs336_data.filtering import *

# Combine the negative and positve labeled training data into one file
# Randomly sample 1400 documents form the negative data since we only have 1400 positive samples to create a balanaced training file
def temp():
    with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\negative_text.txt", "r", encoding="utf-8") as f:
        neg_lines = f.readlines()
    balanced_neg = random.sample(neg_lines, 1400)
    with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\balanced_train.txt", "w", encoding="utf-8") as out:
        with open(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\positive_text.txt", "r", encoding="utf-8") as f:
            out.writelines(f.readlines())
        out.writelines(balanced_neg)

def process_warc_file_to_file(
    warc_path, 
    extract_text_func, 
    output_path, 
    max_docs=1000
):
    count = 0
    with open(warc_path, 'rb') as stream, open(output_path, 'w', encoding='utf-8') as out_file:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            try:
                raw_bytes = record.content_stream().read()
                text = extract_text_func(raw_bytes)
            except Exception as e:
                print(f"[Skipping record] {e}")
                continue

            if not text or not text.strip():
                continue

            out_file.write("----- DOCUMENT START -----\n")
            out_file.write(text.strip() + "\n")
            out_file.write("------ DOCUMENT END ------\n\n")

            count += 1
            if count >= max_docs:
                break


def evaluate_language_id(
    input_path: str,
    language_id_func,
    sample_size: int = 20,
    report_path: str = None
):
    docs = []
    current_doc = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "----- DOCUMENT START -----":
                current_doc = []
            elif line.strip() == "------ DOCUMENT END ------":
                doc_text = "\n".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
            else:
                current_doc.append(line.strip())

    print(f"Found {len(docs)} documents.")

    sample_docs = random.sample(docs, min(sample_size, len(docs)))
    print("\n Sample for manual inspection:\n")

    for i, doc in enumerate(sample_docs):
        lang, score = language_id_func(doc)
        print(f"[Doc {i+1}] Language: {lang}, Confidence: {score:.2f}")
        print(doc[:400], "...\n")
        print("-" * 80)

    english_count = 0
    total = 0
    confidences = []

    for doc in docs:
        lang, score = language_id_func(doc)
        if lang == "en":
            english_count += 1
            confidences.append(score)
        total += 1

    frac_en = english_count / total
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f"Total docs: {total}")
    print(f"English docs: {english_count} ({frac_en:.2%})")
    print(f"Average confidence (English docs): {avg_conf:.2f}")

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Total: {total}\n")
            f.write(f"English: {english_count} ({frac_en:.2%})\n")
            f.write(f"Avg confidence (en): {avg_conf:.2f}\n")


def evaluate_pii_masking_to_file(
    input_path: str,
    mask_func,
    output_path: str,
    sample_size: int = 20
):
    docs = []
    current_doc = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "----- DOCUMENT START -----":
                current_doc = []
            elif line.strip() == "------ DOCUMENT END ------":
                doc_text = "\n".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
            else:
                current_doc.append(line.strip())

    print(f"Found {len(docs)} documents.")

    modified_docs = []
    for doc in docs:
        masked = mask_func(doc)
        if masked != doc:
            modified_docs.append((doc, masked))

    print(f"Found {len(modified_docs)} documents with at least one PII replacement.")

    samples = random.sample(modified_docs, min(sample_size, len(modified_docs)))

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Total documents: {len(docs)}\n")
        out.write(f"Documents with masked PII: {len(modified_docs)}\n")
        out.write(f"Random Sample of {len(samples)} Examples:\n\n")

        for i, (original, masked) in enumerate(samples):
            out.write(f"===== SAMPLE {i+1} =====\n")
            out.write("Original:\n")
            out.write(original.strip() + "\n\n")
            out.write("Masked:\n")
            out.write(masked.strip() + "\n")
            out.write("-" * 60 + "\n\n")

    print(f"Output written to: {output_path}")


def evaluate_harmful_content_to_file(
    input_path: str,
    find_nsfw_func,
    find_toxic_func,
    output_path: str,
    sample_size: int = 20
):
    docs = []
    current_doc = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "----- DOCUMENT START -----":
                current_doc = []
            elif line.strip() == "------ DOCUMENT END ------":
                doc_text = "\n".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
            else:
                current_doc.append(line.strip())

    print(f"Loaded {len(docs)} documents.")

    flagged_docs = []
    total_docs = len(docs)
    harmful_count = 0
    nsfw_scores = []
    toxic_scores = []

    for doc in docs:
        nsfw_label, nsfw_score = find_nsfw_func(doc)
        toxic_label, toxic_score = find_toxic_func(doc)

        if nsfw_label == "nsfw" or toxic_label == "toxic":
            harmful_count += 1
            flagged_docs.append({
                "text": doc,
                "nsfw": (nsfw_label, nsfw_score),
                "toxic": (toxic_label, toxic_score)
            })

        nsfw_scores.append(nsfw_score)
        toxic_scores.append(toxic_score)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Total documents: {total_docs}\n")
        out.write(f"Harmful documents flagged: {harmful_count} ({harmful_count / total_docs:.2%})\n")
        out.write(f"Average NSFW confidence: {sum(nsfw_scores)/len(nsfw_scores):.2f}\n")
        out.write(f"Average Toxicity confidence: {sum(toxic_scores)/len(toxic_scores):.2f}\n\n")

        sample = random.sample(flagged_docs, min(sample_size, len(flagged_docs)))

        out.write(f"Sample of {len(sample)} flagged documents for manual inspection:\n\n")
        for i, doc_info in enumerate(sample):
            out.write(f"===== SAMPLE {i+1} =====\n")
            out.write("NSFW Prediction: {} (confidence: {:.2f})\n".format(*doc_info["nsfw"]))
            out.write("Toxicity Prediction: {} (confidence: {:.2f})\n".format(*doc_info["toxic"]))
            out.write("Content:\n")
            out.write(doc_info["text"][:1000] + "\n")  
            out.write("-" * 70 + "\n\n")

    print(f"Report saved to {output_path}")


def evaluate_quality_filter_sampled(
    input_path: str,
    quality_filter_func,
    output_path: str,
    sample_size: int = 20
):
    docs = []
    current_doc = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "----- DOCUMENT START -----":
                current_doc = []
            elif line.strip() == "------ DOCUMENT END ------":
                doc_text = "\n".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
            else:
                current_doc.append(line.strip())

    print(f"Loaded {len(docs)} documents total.")

    sample = random.sample(docs, min(sample_size, len(docs)))
    print(f"Evaluating quality on {len(sample)} random documents...")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Evaluated {len(sample)} sampled documents with rule-based quality filter.\n\n")

        for i, doc in enumerate(sample):
            passed = quality_filter_func(doc)
            out.write(f"===== SAMPLE {i+1} =====\n")
            out.write(f"Filter Prediction: {'PASS' if passed else 'FAIL'}\n")
            out.write("Text:\n")
            out.write(doc[:1000] + "\n")
            out.write("-" * 70 + "\n\n")

    print(f"Report saved to: {output_path}")


    # Take an input file of text extracted from the wget command on the list of URLS, write them 
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
                continue


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




if __name__ == "__main__":
    #extract_docs(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\SampleWARC.gz",
    #              r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\negative_text.txt")
    # subsample_urls(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\enwiki-20240420-extracted_urls.txt", 
    # r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\subsampled_positive_urls.txt")
    x=0
    # model = train_classifier()
    #process_warc_file_to_file(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\SampleWARC.gz", extract_text, r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\extract_out.txt")
    #evaluate_language_id(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\extract_out.txt", language_id)
    #evaluate_pii_masking_to_file(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\extract_out.txt", mask_PII, r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\mask_out.txt" )
    #evaluate_harmful_content_to_file(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\extract_out.txt", find_NSFW, find_toxic, r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\censor_out.txt")
    #evaluate_quality_filter_sampled(r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\extract_out.txt", gopher_filter, r"C:\Users\cpbsw\s2025-assignment2-data\cs336-data\cs336_data\docs\gopher_out.txt")
