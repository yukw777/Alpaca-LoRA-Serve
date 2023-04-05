import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for Alpaca-LoRA as a chatbot service"
    )
    # Dataset related.
    parser.add_argument(
        "--model_name_or_path",
        help="Hugging Face model name or path",
        type=str,
    )
    parser.add_argument(
        "--model_type", help="Model type", default="alpaca", choices=["alpaca", "baize"]
    )
    parser.add_argument("--peft", help="Use PEFT", action="store_true")
    parser.add_argument("--load_in_8bit", help="Load in 8bit", action="store_true")
    parser.add_argument(
        "--port",
        help="PORT number where the app is served",
        default=6006,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of requests to handle at the same time",
        default=1,
        type=int,
    )
    parser.add_argument("--api_open", help="Open as API", action="store_true")
    parser.add_argument(
        "--share",
        help="Create and share temporary endpoint (useful in Colab env)",
        action="store_true",
    )
    parser.add_argument(
        "--gen_config_path",
        help="path to GenerationConfig file used in batch mode",
        default="configs/gen_config_default.yaml",
        type=str,
    )
    parser.add_argument(
        "--gen_config_summarization_path",
        help="path to GenerationConfig file used in context summarization",
        default="configs/gen_config_summarization.yaml",
        type=str,
    )
    parser.add_argument(
        "--multi_gpu",
        help="Enable multi gpu mode. This will force not to use Int8 but float16, so you need to check if your system has enough GPU memory",
        action="store_true",
    )

    return parser.parse_args()
