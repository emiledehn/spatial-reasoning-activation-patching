import os
# os.environ["HF_HOME"] = 
# os.environ["HF_DATASETS_CACHE"] = 

from dotenv import load_dotenv
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_dissector import ModelDissector
import data

load_dotenv()

path = "../activation-patching/results/patching/setting2/Llama-3.3-70B-Instruct"
os.makedirs(path, exist_ok=True)

# model and tokenizer
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", token=os.getenv("HF_TOKEN")
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))

# experiment settings
system_message = {
    "role": "system",
    "content": """You are an assistant that determines the relative position of Object C with repsect to Object A.
Output must always follow the requested format.
Do no add reasoning only give the result.""",
}
question = "Determine the relative position of Object C with respect to Object A."
output_format = """Output format:
{
    "x": "left | right | none",
    "y": "front | back | none",
    "z": "top | bottom | none"
}
Use "none" if there is no relative displacement on that axis."""
setting = lambda t1, t2: f"Object B is {t1} of Object A. Object C is {t2} of Object B."
# start patching after this many tokens of the message (skip system + question + output format)
message_offset = 70

# x:0 = left:-1/right:1
# y:1 = back:-1/front:1
# z:2 = bottom:-1/top:1
experiments = [
    {
        "clean": {"t1": [0, 0, 0], "t2": [0, 0, 0]},
        "corrupt": {"t1": [0, 0, 0], "t2": [0, 0, 0]},
        "target": 1
    },
    # more experiments ...
]

def get_answer_for_target_axis(target, answer):
    position = "none"
    if target == 0:
        if answer[target] < 0:
            position = "left"
        else:
            position = "right"
    elif target == 1:
        if answer[target] < 0:
            position = "back"
        else:
            position = "front"
    elif target == 2:
        if answer[target] < 0:
            position = "bottom"
        else:
            position = "top"
    return position

for experiment in experiments:    
    print("Running experiment:", experiment)
    # token index of the targeted axis (token used for metrics computation)
    if experiment["target"] == 0:
        token_idx = 10
    elif experiment["target"] == 1:
        token_idx = 17
    elif experiment["target"] == 2:
        token_idx = 24
    
    patch_model = ModelDissector(model, tokenizer)

    # prepare messages
    clean_message = {
        "role": "user",
        "content": f"""{question}
{output_format}
{setting(data.vector_to_direction(experiment["clean"]["t1"]),data.vector_to_direction(experiment["clean"]["t2"]))}""",
    }
    layer_cache, attention_cache, mlp_cache, outputs = patch_model.generate_cache(
        system_message, clean_message
    )
    corrupt_message = {
        "role": "user",
        "content": f"""{question}
{output_format}
{setting(data.vector_to_direction(experiment["corrupt"]["t1"]),data.vector_to_direction(experiment["corrupt"]["t2"]))}""",
    }

    # compute clean and corrupt answer tokens
    clean_c = [x + y for x, y in zip(experiment["clean"]["t1"], experiment["clean"]["t2"])]
    clean_answer = get_answer_for_target_axis(experiment["target"], clean_c)
    corrupt_c = [x + y for x, y in zip(experiment["corrupt"]["t1"], experiment["corrupt"]["t2"])]
    corrupt_answer = get_answer_for_target_axis(experiment["target"], corrupt_c)
    
    clean_token = tokenizer.encode(clean_answer, add_special_tokens=False)
    corrupt_token = tokenizer.encode(corrupt_answer, add_special_tokens=False)

    # start patching
    experiment_name = f"{data.vector_to_direction(experiment["clean"]["t1"])}:{data.vector_to_direction(experiment["corrupt"]["t1"])}-{data.vector_to_direction(experiment["clean"]["t2"])}:{data.vector_to_direction(experiment["corrupt"]["t2"])}#{experiment["target"]}"
    os.makedirs(f"{path}/{experiment_name}", exist_ok=True)
    residual_patching_metrics = patch_model.exploratory_patching(
        module_type="residual",
        system_message=system_message,
        corrupt_message=corrupt_message,
        token_idx=token_idx,
        clean_token=clean_token,
        corrupt_token=corrupt_token,
        message_offset=message_offset
    )
    with open(
        f"{path}/{experiment_name}/residual_patching_metrics.pkl",
        "wb",
    ) as f:
        pickle.dump(residual_patching_metrics, f)    

    attention_patching_metrics = patch_model.exploratory_patching(
        module_type="attention",
        system_message=system_message,
        corrupt_message=corrupt_message,
        token_idx=token_idx,
        clean_token=clean_token,
        corrupt_token=corrupt_token,
        message_offset=message_offset
    )
    with open(
        f"{path}/{experiment_name}/attention_patching_metrics.pkl",
        "wb",
    ) as f:
        pickle.dump(attention_patching_metrics, f)    

    mlp_patching_metrics = patch_model.exploratory_patching(
        module_type="mlp",
        system_message=system_message,
        corrupt_message=corrupt_message,
        token_idx=token_idx,
        clean_token=clean_token,
        corrupt_token=corrupt_token,
        message_offset=message_offset
    )
    with open(
        f"{path}/{experiment_name}/mlp_patching_metrics.pkl",
        "wb",
    ) as f:
        pickle.dump(mlp_patching_metrics, f)    
