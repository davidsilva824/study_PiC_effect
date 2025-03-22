import pandas as pd
from surprisal import AutoHuggingFaceModel

sentences = ["this monster is a rat eater"]

# Load the model
m = AutoHuggingFaceModel.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')
m.to('cuda')  # Optionally move your model to GPU

# Process sentences
[result] =m.surprise(sentences)

print(f"\n {result} \n\n")
print(f"{dir(result)} \n\n")

x = result.surprisals
print(f"{x} \n\n")

print(sum(x[2:4]))

