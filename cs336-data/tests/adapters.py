#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any
from cs336_data.filtering import extract_text, find_NSFW, find_toxic, gopher_filter, label_quality, language_id, mask_email, mask_ipv4, mask_phone
from cs336_data.deduplication import exact_line_dedup, minhash_dedup


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)
    raise NotImplementedError


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_id(text)
    raise NotImplementedError


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)
    raise NotImplementedError


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone(text)
    raise NotImplementedError


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ipv4(text)
    raise NotImplementedError


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return find_NSFW(text)
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return find_toxic(text)
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    return label_quality(text)
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_filter(text)
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_line_dedup(input_files, output_directory)
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_dedup(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
    raise NotImplementedError   