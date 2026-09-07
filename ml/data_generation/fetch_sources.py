import urllib.request
import json
import os
import sys

def fetch_hf_readme():
    url = "https://huggingface.co/datasets/shambhuraje/Indian_Railway_maintance/raw/main/README.md"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8')
            with open("data/raw/hf_indian_railway_maintenance_readme.md", "w", encoding="utf-8") as f:
                f.write(content)
            print("Successfully saved HF README")
    except Exception as e:
        print(f"Error fetching HF README: {e}")

def fetch_hf_sample_header():
    # Fetch first 4KB of CSV to inspect schema
    url = "https://huggingface.co/datasets/shambhuraje/Indian_Railway_maintance/resolve/main/indian_railway_predictive_maintenance_100k.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-4096'})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            lines = content.splitlines()
            with open("data/raw/hf_dataset_sample_header.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(lines[:10]))
            print("Header and sample lines from HF 100k:")
            for l in lines[:5]:
                print("  ", l)
    except Exception as e:
        print(f"Error fetching HF sample: {e}")

if __name__ == '__main__':
    fetch_hf_readme()
    fetch_hf_sample_header()
