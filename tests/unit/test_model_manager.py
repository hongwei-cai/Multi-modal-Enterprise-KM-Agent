from unittest.mock import MagicMock, patch

from src.rag.managers.model_manager import (
    ABTestConfig,
    ModelConfig,
    ModelManager,
    ModelTier,
)


def test_model_versioning():
    """Test model versioning functionality."""
    with patch(
        "src.rag.managers.model_manager.ModelManager._load_model_versions"
    ), patch("pathlib.Path.mkdir"), patch(
        "pathlib.Path.glob", return_value=[]
    ), patch(  # Mock to avoid loading files
        "src.rag.managers.model_manager.MLflowExperimentTracker"
    ):
        manager = ModelManager()

        config = ModelConfig(
            "test-model", ModelTier.BALANCED, 2.0, 100, 0.8, "Test model description"
        )
        manager.save_model_version("v1.0", config)

        assert "v1.0" in manager.list_model_versions()
        loaded_config = manager.get_model_config("v1.0")
        assert loaded_config.name == "test-model"  # Changed from model_name to name


def test_model_switching():
    """Test seamless model switching."""
    with patch(
        "src.rag.managers.model_manager.ModelManager._load_model_versions"
    ), patch("pathlib.Path.mkdir"), patch("pathlib.Path.glob", return_value=[]), patch(
        "src.rag.managers.model_manager.MLflowExperimentTracker"
    ):
        manager = ModelManager()

        # Mock models
        config1 = ModelConfig("model1", ModelTier.SPEED, 1.0, 50, 0.6, "Speed model")
        config2 = ModelConfig(
            "model2", ModelTier.QUALITY, 4.0, 200, 0.9, "Quality model"
        )

        manager.save_model_version("speed", config1)
        manager.save_model_version("quality", config2)

        # Mock load_model_with_fallback
        with patch.object(manager, "load_model_with_fallback") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())

            assert manager.switch_model("speed")
            assert manager.current_model == "speed"

            assert manager.switch_model("quality")
            assert manager.current_model == "quality"


def test_ab_testing():
    """Test A/B testing framework."""
    with patch(
        "src.rag.managers.model_manager.ModelManager._load_model_versions"
    ), patch("pathlib.Path.mkdir"), patch("pathlib.Path.glob", return_value=[]), patch(
        "src.rag.managers.model_manager.MLflowExperimentTracker"
    ):
        manager = ModelManager()

        config = ABTestConfig("test_ab", "model_a", "model_b", 0.7)
        manager.start_ab_test(config)

        # Test traffic split (deterministic for testing)
        import random

        random.seed(42)  # For consistent testing
        model = manager.get_ab_test_model("test_ab")
        assert model in ["model_a", "model_b"]

        # Record result
        manager.record_ab_test_result("test_ab", model, {"latency": 100})
