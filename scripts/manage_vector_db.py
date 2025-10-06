#!/usr/bin/env python3
"""
Vector Database Management Script.
This script provides utilities for managing the
ChromaDB vector database used by the KM Agent.

Usage:
    python scripts/manage_vector_db.py stats          # Show database statistics
    python scripts/manage_vector_db.py list           # List all collections
    python scripts/manage_vector_db.py clear          # Clear all data
    python scripts/manage_vector_db.py backup <file>  # Backup database to file
    python scripts/manage_vector_db.py restore <file> # Restore database from file
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from src.rag.vector_database import get_vector_db


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("vector_db_management.log"),
        ],
    )


def show_stats(vector_db):
    """Show database statistics."""
    logger = logging.getLogger(__name__)

    try:
        # Get collection info
        collections = vector_db.client.list_collections()
        collections = [col.name for col in collections]  # Convert to list of names
        logger.info("Database Statistics")
        logger.info("=" * 50)

        if not collections:
            logger.info("No collections found")
            return

        total_documents = 0
        total_embeddings = 0

        for collection_name in collections:
            try:
                collection = vector_db.get_collection(collection_name)
                count = collection.count()
                total_documents += count
                total_embeddings += count

                logger.info("Collection: %s", collection_name)
                logger.info("  Documents: %d", count)

                # Try to get metadata if available
                try:
                    # This is ChromaDB specific - may not work with all implementations
                    metadata = collection.metadata
                    if metadata:
                        logger.info("  Metadata: %s", metadata)
                except (AttributeError, ValueError):
                    pass

                logger.info("")

            except (RuntimeError, ValueError, OSError) as e:
                logger.error(
                    "Error getting info for collection %s: %s", collection_name, e
                )

        logger.info("Summary:")
        logger.info("  Total collections: %d", len(collections))
        logger.info("  Total documents: %d", total_documents)
        logger.info("  Total embeddings: %d", total_embeddings)

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error getting database statistics: %s", e)


def list_collections(vector_db):
    """List all collections in the database."""
    logger = logging.getLogger(__name__)

    try:
        collections = vector_db.client.list_collections()
        collections = [col.name for col in collections]
        logger.info("Collections in database:")
        logger.info("=" * 30)

        if not collections:
            logger.info("No collections found")
            return

        for i, collection_name in enumerate(collections, 1):
            logger.info("%d. %s", i, collection_name)

        logger.info("\nTotal: %d collections", len(collections))

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error listing collections: %s", e)


def clear_database(vector_db, force: bool = False):
    """Clear all data from the database."""
    logger = logging.getLogger(__name__)

    try:
        collections = vector_db.client.list_collections()
        collections = [col.name for col in collections]

        if not collections:
            logger.info("Database is already empty")
            return

        if not force:
            # Show confirmation prompt
            print(
                f"This will delete {len(collections)} collections with all their data."
            )
            response = input("Are you sure? Type 'yes' to confirm: ")
            if response.lower() != "yes":
                logger.info("Operation cancelled")
                return

        logger.info("Clearing database...")

        for collection_name in collections:
            try:
                vector_db.delete_collection(collection_name)
                logger.info("Deleted collection: %s", collection_name)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error("Error deleting collection %s: %s", collection_name, e)

        logger.info("Database cleared successfully")

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error clearing database: %s", e)


def backup_database(vector_db, backup_file: str):
    """Backup database to a file."""
    logger = logging.getLogger(__name__)

    try:
        # For ChromaDB, we need to backup the persist directory
        # This is a simple implementation - \
        # in production you'd want more sophisticated backup

        # Get the persist directory from the vector db config
        # This assumes ChromaDB with persistence
        persist_dir = getattr(vector_db, "persist_directory", None)

        if not persist_dir:
            logger.error("Cannot determine database location for backup")
            return

        persist_path = Path(persist_dir)
        if not persist_path.exists():
            logger.error("Database directory does not exist: %s", persist_path)
            return

        backup_path = Path(backup_file)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Backing up database from %s to %s", persist_path, backup_path)

        # Create zip archive
        shutil.make_archive(str(backup_path.with_suffix("")), "zip", persist_path)

        logger.info("Backup completed successfully: %s.zip", backup_path)

    except (RuntimeError, ValueError, OSError, shutil.Error) as e:
        logger.error("Error creating backup: %s", e)


def restore_database(vector_db, backup_file: str):
    """Restore database from a backup file."""
    logger = logging.getLogger(__name__)

    try:
        backup_path = Path(backup_file)
        if not backup_path.exists():
            logger.error("Backup file does not exist: %s", backup_path)
            return

        # For ChromaDB, we need to restore to the persist directory
        persist_dir = getattr(vector_db, "persist_directory", None)

        if not persist_dir:
            logger.error("Cannot determine database location for restore")
            return

        persist_path = Path(persist_dir)

        # Warn about overwriting
        if persist_path.exists():
            response = input(
                f"This will overwrite existing data in {persist_path}. \
                    Continue? (yes/no): "
            )
            if response.lower() != "yes":
                logger.info("Restore cancelled")
                return

        logger.info("Restoring database from %s to %s", backup_path, persist_path)

        # Extract zip archive
        if backup_path.suffix == ".zip":
            shutil.unpack_archive(backup_path, persist_path.parent, "zip")
        else:
            # Assume it's a directory
            if backup_path.is_dir():
                shutil.copytree(backup_path, persist_path, dirs_exist_ok=True)
            else:
                logger.error(
                    "Unsupported backup format. Expected .zip file or directory"
                )
                return

        logger.info("Restore completed successfully")

        # Note: May need to restart the application for changes to take effect
        logger.warning(
            "You may need to restart the application for restored data to be available"
        )

    except (RuntimeError, ValueError, OSError, shutil.Error) as e:
        logger.error("Error restoring backup: %s", e)


def export_data(vector_db, output_file: str, collection_name: Optional[str] = None):
    """Export collection data to JSON."""
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
            logger.info("No collections to export")
            return

        data_to_export = {}

        for coll_name in collections:
            try:
                collection = vector_db.get_collection(coll_name)

                # Get all documents, metadatas, and IDs
                results = collection.get(include=["documents", "metadatas"])

                data_to_export[coll_name] = {
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

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_export, f, indent=2, ensure_ascii=False)

        logger.info("Data exported to: %s", output_file)

    except (RuntimeError, ValueError, OSError, IOError) as e:
        logger.error("Error exporting data: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Manage the vector database")
    parser.add_argument(
        "command",
        choices=["stats", "list", "clear", "backup", "restore", "export"],
        help="Command to execute",
    )
    parser.add_argument(
        "--collection", "-c", help="Specific collection for export command"
    )
    parser.add_argument(
        "--file", "-f", help="File path for backup/restore/export commands"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize vector database
        logger.info("Initializing vector database...")
        vector_db = get_vector_db()

        # Execute command
        if args.command == "stats":
            show_stats(vector_db)
        elif args.command == "list":
            list_collections(vector_db)
        elif args.command == "clear":
            clear_database(vector_db, args.force)
        elif args.command == "backup":
            if not args.file:
                logger.error("Backup command requires --file argument")
                sys.exit(1)
            backup_database(vector_db, args.file)
        elif args.command == "restore":
            if not args.file:
                logger.error("Restore command requires --file argument")
                sys.exit(1)
            restore_database(vector_db, args.file)
        elif args.command == "export":
            if not args.file:
                logger.error("Export command requires --file argument")
                sys.exit(1)
            export_data(vector_db, args.file, args.collection)

    except (RuntimeError, ValueError, OSError, ImportError) as e:
        logger.error("Command failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
