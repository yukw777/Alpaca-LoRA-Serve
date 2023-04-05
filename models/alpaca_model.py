from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name_or_path, peft, load_in_8bit):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if peft:
        config = PeftConfig.from_pretrained(model_name_or_path)
        # we need to set device_map={"":0} due to the following issue:
        # https://github.com/tloen/alpaca-lora/issues/14#issuecomment-1471263165
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(model, model_name_or_path, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_8bit=load_in_8bit, device_map="auto"
        )
    return model, tokenizer
