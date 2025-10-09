"""CLI to generate QA pairs from documents using the RAG QA pair generator."""
import argparse
import logging
from pathlib import Path

from src.rag.components.qa_pair_generator import get_qa_generator


def find_documents(input_dir: Path):
    exts = [".txt", ".pdf"]
    files = [str(p) for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, help="Directory with documents (pdf/txt)"
    )
    parser.add_argument("--output", default="data/processed/qa_pairs.jsonl")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = find_documents(input_dir)
    if not files:
        raise SystemExit(f"No supported documents found in {input_dir}")

    gen = get_qa_generator()
    out = gen.generate_from_files(
        files,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_pairs=args.max_pairs,
        output_path=args.output,
    )

    print(f"Saved QA pairs to: {out}")


if __name__ == "__main__":
    main()
