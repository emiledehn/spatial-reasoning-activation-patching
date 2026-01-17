# Investigating the Internal Representations of Spatial Reasoning in Large Language Models Using Activation Patching

## Dataset
Contains code to create the settings for the questions.
## Inference
Contains code to test the model and visualize results.
## Patching
Contains code to patch the model and visualize results.

## Patching Prompt
### System 
You are an assistant that determines the relative position of Object C with respect to Object A.\
Output must always follow the requested format.\
Do not add reasoning only give the result.
### User
Determine the relative position of Object C with respect to Object A.\
Output format:\
{\
    "x": "left | right | none",\
    "y": "front | back | none",\
    "z": "top | bottom | none"\
}\
Use "none" if there is no relative displacement on that axis.\
Object B is T1 of Object A. Object C is T2 of Object B.

### Additional Questions 
(Adjust prompt template)
- Which Objects are or which object is the farthest to the left/right/back/front/bottom/top?
- Imagine the Objects are boxes of the same size and are placed in a way that they either touch with a face, edge or corner or are placed within each other. Do boxes C and A touch at a face, edge, corner or is box C located inside box A?
- Is Object B or Object C closer/farther to/from Object A?
- Which Objects are closest/farthest to ech other?