import ast
import re

def clean_prediction(prediction):
    # remove markdown code blocks
    prediction = re.sub(r"```(?:json|python|text)?\s*", "", prediction)
    prediction = re.sub(r"```\s*", "", prediction)

    # remove instruct specific tokens
    prediction = prediction.replace("assistant", "")

    # convert JSON booleans to Python booleans
    prediction = prediction.replace("true", "True").replace("false", "False")

    # convert ' to ""
    prediction = prediction.replace("'", '"')

    prediction = prediction.strip()
    data = ast.literal_eval(prediction)
    return data

def validate_format(prediction):
    valid_values = {
        "x": ["left", "right", "none"],
        "y": ["front", "back", "none"],
        "z": ["top", "bottom", "none"]
    }
    try:
        data = clean_prediction(prediction)

        if not isinstance(data, dict):
            print("Output is not a dictionary.")
            return False, prediction

        actual_keys = set(data.keys())
        if actual_keys != valid_values.keys():
            print(f"Invalid keys: {actual_keys}, expected: {valid_values.keys()}")
            return False, prediction

        for key in valid_values.keys():
            actual_value = data[key]
            if actual_value not in valid_values[key]:
                print(f"Invalid value {actual_value}, expected: {valid_values[key]}")
                return False, prediction
    except Exception as e:
        print(f"JSON parsing error: {e, prediction}")
        return False, prediction
    return True, data

def validate_prediction(prediction, t1, t2, c):
    truth = {
        "x": "none",
        "y": "none",
        "z": "none",
    }
    if c[2] < 0:
        truth["z"] = "bottom"
    elif c[2] > 0:
        truth["z"] = "top"
    if c[1] < 0:
        truth["y"] = "back"
    elif c[1] > 0:
        truth["y"] = "front"
    if c[0] < 0:
        truth["x"] = "left"
    elif c[0] > 0:
        truth["x"] = "right"
    return prediction == truth, truth
