from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_model_manager_disk_operations():
    """Mock ModelManager disk operations to prevent JSON loading errors in tests."""
    with patch("src.rag.model_manager.ModelManager._load_model_versions"), patch(
        "pathlib.Path.mkdir"
    ), patch("pathlib.Path.glob", return_value=[]):
        yield
