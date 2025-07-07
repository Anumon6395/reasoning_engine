import argparse
import os
from src.reasoning import gauss_seidel_chain

def main():
    parser = argparse.ArgumentParser(description='Run Gauss-Seidel chain search in the problem database.')
    parser.add_argument('--file', type=str, help='Path to query text file.')
    parser.add_argument('--text', type=str, help='Query as a string.')
    parser.add_argument('--max-iter', type=int, default=100, help='Maximum number of iterations.')
    parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for diff norm to stop.')
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

    chain = gauss_seidel_chain(query, max_iter=args.max_iter, tol=args.tol)
    print(f"Gauss-Seidel chain (max_iter={args.max_iter}, tol={args.tol}):")
    for step in chain:
        print(f"Step {step['step']}: Index {step['index']}, Distance {step['distance']:.4f}, Diff Norm {step['diff_norm']:.4f}")
        print(f"    File: {step['file']}")
        print(f"    Text: {step['text']}")
        print()
    print(f"Chain length: {len(chain)}")

if __name__ == '__main__':
    main() 