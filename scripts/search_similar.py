# Placeholder for search_similar CLI 

import argparse
import os
from src.search import search_similar

def main():
    parser = argparse.ArgumentParser(description='Search for similar problems in the database.')
    parser.add_argument('--file', type=str, help='Path to query text file.')
    parser.add_argument('--text', type=str, help='Query as a string.')
    parser.add_argument('--topk', type=int, default=1, help='Number of top results to return.')
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        with open(args.file, 'r') as f:
            query = f.read().strip()
    elif args.text:
        query = args.text
    else:
        print("Please provide a query with --file or --text.")
        return

    results = search_similar(query, top_k=args.topk)
    print(f"Top {args.topk} similar problems:")
    for i, res in enumerate(results):
        print(f"[{i+1}] Index: {res['index']}, Distance: {res['distance']:.4f}")
        print(f"    File: {res['file']}")
        print(f"    Text: {res['text']}")
        print()

if __name__ == '__main__':
    main() 