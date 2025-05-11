import logging
from lora_finetune.ui import build_ui

logging.basicConfig(level=logging.INFO)

def main():
    demo = build_ui()
    demo.launch()

if __name__ == "__main__":
    main() 