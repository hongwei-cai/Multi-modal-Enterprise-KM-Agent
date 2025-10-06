#!/usr/bin/env python3
"""
Data Export/Import Utility Script.
This script provides utilities for exporting and importing data in various formats.

Usage:
    python scripts/data_export_import.py \
        export --format json --output data.json
    python scripts/data_export_import.py \
        export --format csv --output data.csv --collection docs
    python scripts/data_export_import.py \
        import --format json --input data.json
    python scripts/data_export_import.py \
        convert --from json --to csv --input data.json --output data.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.rag.vector_database import get_vector_db


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data_export_import.log"),
        ],
    )


def export_to_json(
    vector_db, output_file: str, collection_name: Optional[str] = None
) -> bool:
    """Export data to JSON format."""
    logger = logging.getLogger(__name__)

    try:
        collections = (
            [collection_name]
            if collection_name
            else vector_db.client.list_collections()
        )
        collections = (
            [col.name for col in collections] if not collection_name else collections
        )

        if not collections:
            logger.warning("No collections to export")
            return False

        export_data = {}

        for coll_name in collections:
            try:
                collection = vector_db.get_collection(coll_name)
                results = collection.get(include=["documents", "metadatas"])

                export_data[coll_name] = {
                    "documents": results.get("documents", []),
                    "metadatas": results.get("metadatas", []),
                    "ids": results.get("ids", []),
                }

                logger.info(
                    "Exported collection: %s (%d documents)",
                    coll_name,
                    len(results.get("documents", [])),
                )

            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Error exporting collection %s: %s", coll_name, e)
                return False

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info("Data exported to JSON: %s", output_file)
        return True

    except (RuntimeError, ValueError, OSError, IOError) as e:
        logger.error("Error exporting to JSON: %s", e)
        return False


def export_to_csv(
    vector_db, output_file: str, collection_name: Optional[str] = None
) -> bool:
    """Export data to CSV format."""
    logger = logging.getLogger(__name__)

    try:
        collections = (
            [collection_name]
            if collection_name
            else vector_db.client.list_collections()
        )
        collections = (
            [col.name for col in collections] if not collection_name else collections
        )

        if not collections:
            logger.warning("No collections to export")
            return False

        # For CSV, we'll flatten the data
        all_rows = []

        for coll_name in collections:
            try:
                collection = vector_db.get_collection(coll_name)
                results = collection.get(include=["documents", "metadatas"])

                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
                ids = results.get("ids", [])

                for i, doc in enumerate(documents):
                    row = {
                        "collection": coll_name,
                        "id": ids[i] if i < len(ids) else "",
                        "document": doc,
                    }

                    # Add metadata fields
                    if i < len(metadatas) and metadatas[i]:
                        for key, value in metadatas[i].items():
                            row[f"metadata_{key}"] = str(value)

                    all_rows.append(row)

                logger.info(
                    "Processed collection: %s (%d documents)", coll_name, len(documents)
                )

            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Error processing collection %s: %s", coll_name, e)
                return False

        if not all_rows:
            logger.warning("No data to export")
            return False

        # Write to CSV
        fieldnames: Set[str] = set()
        for row in all_rows:
            fieldnames.update(row.keys())

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(all_rows)

        logger.info("Data exported to CSV: %s (%d rows)", output_file, len(all_rows))
        return True

    except (RuntimeError, ValueError, OSError, IOError) as e:
        logger.error("Error exporting to CSV: %s", e)
        return False


def import_from_json(
    vector_db, input_file: str, collection_name: Optional[str] = None
) -> bool:
    """Import data from JSON format."""
    logger = logging.getLogger(__name__)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error("Invalid JSON format: expected object at root")
            return False

        for coll_name, coll_data in data.items():
            if collection_name and coll_name != collection_name:
                continue

            try:
                # Validate data structure
                if not isinstance(coll_data, dict):
                    logger.error("Invalid data structure for collection %s", coll_name)
                    continue

                documents = coll_data.get("documents", [])
                metadatas = coll_data.get("metadatas", [])
                ids = coll_data.get("ids", [])

                if len(documents) != len(metadatas) or len(documents) != len(ids):
                    logger.error("Mismatched array lengths in collection %s", coll_name)
                    continue

                # Get or create collection
                try:
                    collection = vector_db.get_collection(coll_name)
                except ValueError:
                    # Collection doesn't exist, create it
                    collection = vector_db.create_collection(coll_name)

                # Add documents
                if documents:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas if metadatas else None,
                        ids=ids,
                    )

                logger.info(
                    "Imported collection: %s (%d documents)", coll_name, len(documents)
                )

            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Error importing collection %s: %s", coll_name, e)
                return False

        logger.info("Data imported from JSON: %s", input_file)
        return True

    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.error("Error reading JSON file: %s", e)
        return False


def convert_formats(
    input_file: str, input_format: str, output_file: str, output_format: str
) -> bool:
    """Convert between different data formats without using the vector database."""
    logger = logging.getLogger(__name__)

    try:
        # Load input data
        if input_format == "json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            logger.error("Unsupported input format: %s", input_format)
            return False

        # Convert to output format
        if output_format == "csv":
            return _convert_json_to_csv(data, output_file)
        elif output_format == "json":
            # Already in JSON, just copy
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Data converted to JSON: %s", output_file)
            return True
        else:
            logger.error("Unsupported output format: %s", output_format)
            return False

    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.error("Error converting formats: %s", e)
        return False


def _convert_json_to_csv(data: Dict[str, Any], output_file: str) -> bool:
    """Convert JSON data structure to CSV."""
    logger = logging.getLogger(__name__)

    try:
        all_rows = []

        for coll_name, coll_data in data.items():
            documents = coll_data.get("documents", [])
            metadatas = coll_data.get("metadatas", [])
            ids = coll_data.get("ids", [])

            for i, doc in enumerate(documents):
                row = {
                    "collection": coll_name,
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                }

                # Add metadata fields
                if i < len(metadatas) and metadatas[i]:
                    for key, value in metadatas[i].items():
                        row[f"metadata_{key}"] = str(value)

                all_rows.append(row)

        if not all_rows:
            logger.warning("No data to convert")
            return False

        # Write to CSV
        fieldnames: Set[str] = set()
        for row in all_rows:
            fieldnames.update(row.keys())

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(all_rows)

        logger.info("Data converted to CSV: %s (%d rows)", output_file, len(all_rows))
        return True

    except (IOError, OSError) as e:
        logger.error("Error writing CSV: %s", e)
        return False


def validate_files(
    input_file: Optional[str] = None, output_file: Optional[str] = None
) -> bool:
    """Validate input and output files."""
    logger = logging.getLogger(__name__)

    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error("Input file does not exist: %s", input_file)
            return False
        if not input_path.is_file():
            logger.error("Input path is not a file: %s", input_file)
            return False

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if output file exists and warn
        if output_path.exists():
            response = input(
                f"Output file {output_file} already exists. Overwrite? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                logger.info("Operation cancelled")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export and import data in various formats"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export data from vector database"
    )
    export_parser.add_argument(
        "--format", choices=["json", "csv"], required=True, help="Export format"
    )
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")
    export_parser.add_argument(
        "--collection", "-c", help="Specific collection to export (optional)"
    )

    # Import command
    import_parser = subparsers.add_parser(
        "import", help="Import data to vector database"
    )
    import_parser.add_argument(
        "--format", choices=["json"], required=True, help="Import format"
    )
    import_parser.add_argument("--input", "-i", required=True, help="Input file path")
    import_parser.add_argument(
        "--collection", "-c", help="Specific collection to import (optional)"
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert between data formats"
    )
    convert_parser.add_argument(
        "--from",
        dest="from_format",
        choices=["json"],
        required=True,
        help="Input format",
    )
    convert_parser.add_argument(
        "--to",
        dest="to_format",
        choices=["json", "csv"],
        required=True,
        help="Output format",
    )
    convert_parser.add_argument("--input", "-i", required=True, help="Input file path")
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Validate files
        if args.command in ["import", "convert"]:
            if not validate_files(
                input_file=getattr(args, "input", None),
                output_file=getattr(args, "output", None),
            ):
                sys.exit(1)
        elif args.command == "export":
            if not validate_files(output_file=args.output):
                sys.exit(1)

        success = False

        if args.command == "export":
            # Initialize vector database for export
            logger.info("Initializing vector database...")
            vector_db = get_vector_db()

            if args.format == "json":
                success = export_to_json(vector_db, args.output, args.collection)
            elif args.format == "csv":
                success = export_to_csv(vector_db, args.output, args.collection)

        elif args.command == "import":
            # Initialize vector database for import
            logger.info("Initializing vector database...")
            vector_db = get_vector_db()

            if args.format == "json":
                success = import_from_json(vector_db, args.input, args.collection)

        elif args.command == "convert":
            success = convert_formats(
                args.input, args.from_format, args.output, args.to_format
            )

        if success:
            logger.info("Operation completed successfully")
        else:
            logger.error("Operation failed")
            sys.exit(1)

    except (RuntimeError, ValueError, OSError, ImportError) as e:
        logger.error("Command failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
