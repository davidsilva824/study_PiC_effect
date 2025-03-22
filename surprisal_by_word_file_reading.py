import pandas as pd
from surprisal import AutoHuggingFaceModel

# Read sentences from a CSV file (assuming one sentence per line)
with open("list_non_head_all.csv", "r") as file:
    sentences = [line.strip() for line in file.readlines()]

# Load the model
m = AutoHuggingFaceModel.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')
m.to('cuda')  # Move model to GPU

# Process sentences
for result in m.surprise(sentences):  # Correct input format
    print(f"\n {result} \n\n")
    #print(f"{dir(result)} \n\n")

    x = result.surprisals  # Extract surprisals
    print(f"{x} \n\n")

    print(sum(x[2:4]))  # 