import pytest
import app

def test_imports():
    assert hasattr(app, 'upload_images')
    assert hasattr(app, 'auto_caption_images')
    assert hasattr(app, 'train_lora')
    assert hasattr(app, 'generate_image')

def test_gradio_blocks():
    assert hasattr(app, 'demo') 