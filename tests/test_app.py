"""
Tests for the LoRA fine-tuning application.
"""

import pytest
from lora_finetune import data, training, inference, utils, ui


def test_utils_imports():
    """Test that utils module has all required functions."""
    assert hasattr(utils, "get_model_zoo")
    assert hasattr(utils, "get_lora_gallery")
    assert hasattr(utils, "lora_path")
    assert hasattr(utils, "model_path")


def test_data_imports():
    """Test that data module has all required functions."""
    assert hasattr(data, "upload_images")
    assert hasattr(data, "auto_caption_images")


def test_training_imports():
    """Test that training module has all required functions."""
    assert hasattr(training, "train_lora")


def test_inference_imports():
    """Test that inference module has all required functions."""
    assert hasattr(inference, "generate_image")


def test_ui_imports():
    """Test that UI module has build_ui function."""
    assert hasattr(ui, "build_ui")


def test_model_zoo():
    """Test that model zoo returns valid data."""
    models = utils.get_model_zoo()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(m, str) for m in models)


def test_lora_gallery():
    """Test that LoRA gallery returns valid data."""
    loras = utils.get_lora_gallery()
    assert isinstance(loras, list)
    assert len(loras) > 0
    assert all(isinstance(l, str) for l in loras)


def test_model_path():
    """Test model path resolution."""
    # Should return the HF path for known models
    assert "runwayml" in utils.model_path("Stable Diffusion v1.5")
    # Should return the input for unknown models
    assert utils.model_path("unknown-model") == "unknown-model"


def test_lora_path():
    """Test LoRA path resolution."""
    # Should return the HF path for known LoRAs
    assert "arkayoji" in utils.lora_path("Waifu LoRA")
    # Should return the input for unknown LoRAs
    assert utils.lora_path("unknown-lora") == "unknown-lora"


def test_train_lora_command_generation():
    """Test that train_lora generates a valid command."""
    command, status = training.train_lora(
        dataset_dir="/tmp/dataset", base_model="test-model", output_dir="/tmp/output"
    )
    assert isinstance(command, str)
    assert isinstance(status, str)
    assert "accelerate launch" in command
    assert "test-model" in command
    assert "/tmp/dataset" in command
    assert "/tmp/output" in command
