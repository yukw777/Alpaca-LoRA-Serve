from models.alpaca_model import load_model
from gens.stream_gen import StreamModel

from miscs.utils import get_generation_config


def initialize_globals(args):
    global model, stream_model, tokenizer
    global generation_config, gen_config_summarization
    global model_type, batch_enabled

    model_type = args.model_type
    batch_enabled = True if args.batch_size > 1 else False

    model, tokenizer = load_model(
        model_name_or_path=args.model_name_or_path,
        peft=args.peft,
        load_in_8bit=args.load_in_8bit,
    )

    generation_config = get_generation_config(args.gen_config_path)
    gen_config_summarization = get_generation_config(args.gen_config_summarization_path)

    if not batch_enabled:
        stream_model = StreamModel(model, tokenizer)
