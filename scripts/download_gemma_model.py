#!/usr/bin/env python3
"""
Download the officially supported Gemma 4 model variant for local testing.
Competition model: google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct
Fast local testing variants:
 - google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct
 - google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct
 - google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct
"""

import argparse
import sys
import kagglehub

DEFAULT_MODEL = "google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct"

def main():
    parser = argparse.ArgumentParser(description="Download Gemma 4 model using KaggleHub")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model handle to download (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    print(f"Downloading model {args.model} via kagglehub...")
    path = kagglehub.model_download(args.model)
    print(f"Model downloaded successfully to: {path}")

if __name__ == "__main__":
    main()
