import os
# os.environ["HF_HOME"] = 
# os.environ["HF_DATASETS_CACHE"] =

import data
from dotenv import load_dotenv
import numpy as np
import pickle
import validation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

experiment_name = "inference_experiment"
path = "../inference_results"
models = [
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B-Instruct"),
]
setting = lambda t1, t2: f"Object B is {t1} of Object A. Object C is {t2} of Object B."
system_context = """You are an assistant that determines the relative position of Object C with respect to Object A.
Output must always follow the requested format.
Do not add reasoning only give the result."""
question = "Determine the relative position of Object C with respect to Object A."
output_format = """Output format:
{
    "x": "left | right | none",
    "y": "front | back | none",
    "z": "top | bottom | none"
}
Use "none" if there is no relative displacement on that axis."""

prompt = lambda t1, t2: [
    {"role": "system", "content": system_context},
    {
        "role": "user",
        "content": f"""{question}
{output_format}
{setting(t1, t2)}""",
    },
]

for model_id, model_name in models:
    logfile = f"{path}/{experiment_name}/{model_name}"
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    log = {
        "model": model_name,
        "setting": setting("placeholder", "placeholder"),
        "system_context": system_context,
        "question": question,
        "output_format": output_format,
        "output": [],
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", token=os.getenv("HF_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
  
    positions = [
        position
        for position in data.generate_positions()
        # not all objects can share the same position and c cannot at origin
        if (np.sum(np.abs(position[0])) != 0 or np.sum(np.abs(position[1])) != 0) and np.sum(np.abs(position[2])) != 0
    ]

    for t1, t2, c in positions:
        inputs = tokenizer.apply_chat_template(
            prompt(data.vector_to_direction(t1), data.vector_to_direction(t2)),
            return_tensors="pt",
        ).to(model.device)

        decoded_inputs = tokenizer.decode(inputs[0], skip_special_tokens=False)

        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=128,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                return_dict_in_generate=True,
            )

        response = tokenizer.decode(
            outputs.sequences[:, inputs.shape[1] :][0], skip_special_tokens=True
        )

        valid_format, prediction = validation.validate_format(response)
        valid_answer, truth = validation.validate_prediction(prediction, t1, t2, c)

        log_entry = {
            "t1": {
                "vector": t1,
                "direction": data.vector_to_direction(t1),
            },
            "t2": {
                "vector": t2,
                "direction": data.vector_to_direction(t2),
            },
            "c": {"vector": c, "direction": data.vector_to_direction(c)},
            "valid_format": valid_format,
            "valid_answer": valid_answer,
            "prediction": prediction,
            "truth": truth,
        }
        log["output"].append(log_entry)

    with open(f"{logfile}.pkl", "wb") as f:
        pickle.dump(log, f)
