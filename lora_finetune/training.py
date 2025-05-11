import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def train_lora(dataset_dir: str, base_model: str, output_dir: str) -> Tuple[str, str]:
    """
    Generate a command to train a LoRA model using the specified dataset and base model.
    Args:
        dataset_dir: Path to dataset
        base_model: Model identifier (e.g., 'runwayml/stable-diffusion-v1-5')
        output_dir: Output directory for LoRA weights
    Returns:
        command: Training command string
        status: Status message
    """
    command = (
        f"accelerate launch --mixed_precision=\"fp16\" sd.py "
        f"--pretrained_model_name_or_path={base_model} "
        f"--train_data_dir={dataset_dir} --output_dir={output_dir} "
        f"--resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 "
        f"--max_train_steps=15000 --learning_rate=1e-04 --checkpointing_steps=500 "
        f'--validation_prompt="a woman wearing red lipstick with black hair" --seed=1337'
    )
    logger.info(f"Generated training command: {command}")
    return command, "Training command generated. Run this in your terminal to start training." 