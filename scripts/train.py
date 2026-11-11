"""Launch LoRA or QLoRA training. OLMo 7B refuses to start on CPU."""

from myna_ai.cli import train

if __name__ == "__main__":
    train()
