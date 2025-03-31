import pandas as pd
from surprisal import AutoHuggingFaceModel

sentences = ["a bird was flying in the sky"]

model = 'PatrickHaller/hgrn2_pile_100m_distill_babylm'
# Load the model
m = AutoHuggingFaceModel.from_pretrained(model,  model_class="causal", trust_remote_code=True )
m.to('cuda')  # Optionally move your model to GPU

# Process sentences
[result] =m.surprise(sentences)

print(f"\n {result} \n\n")
print(f"{dir(result)} \n\n")

x = result.surprisals
print(f"{x} \n\n")

print(sum(x[2:4]))

