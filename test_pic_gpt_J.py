import pandas as pd
from surprisal import AutoHuggingFaceModel

# File paths
head_file = "list_head.csv"
regular_non_head_file = "list_non_head_regular.csv"
irregular_non_head_file = "list_non_head_irregular.csv"
pluralia_tantum_file = "list_non_head_pluralia_tantum.csv"

# Load model
m = AutoHuggingFaceModel.from_pretrained('EleutherAI/gpt-j-6B')
m.to('cuda')  # Optionally move model to GPU

# Load words
with open(head_file, "r") as f:
    heads = [line.strip() for line in f.readlines()]

with open(regular_non_head_file, "r") as f:
    regular_non_heads = [line.strip() for line in f.readlines()]

with open(irregular_non_head_file, "r") as f:
    irregular_non_heads = [line.strip() for line in f.readlines()]

with open(pluralia_tantum_file, "r") as f:
    pluralia_tantum = [line.strip() for line in f.readlines()]

data = []

# Function to compute and print surprisal with all the tokenization possibilities
def compound_combination(category_name, non_heads):
    for head in heads:
        for non_head in non_heads:
            sentence = f"{non_head} {head}"
            [result] = m.surprise([sentence])
            print(result)  # Debugging print

            surprisal_values = result.surprisals
            print(surprisal_values)  # Debugging print

            token_count = len(surprisal_values)
            
            if token_count == 5:
                surprisal_non_head = sum(surprisal_values[0:3])
                surprisal_head = sum(surprisal_values[3:5])
            elif token_count == 4:
                surprisal_non_head = sum(surprisal_values[0:2])
                surprisal_head = sum(surprisal_values[2:4])

            elif token_count == 3:
                if  result.tokens[0] == non_head:  # Pluralia Tantum case (non-head is split into two tokens)
                    surprisal_non_head = surprisal_values[0]
                    surprisal_head = sum(surprisal_values[1:3])
                else:
                    surprisal_non_head = sum(surprisal_values[0:2])
                    surprisal_head = surprisal_values[2]

            elif token_count == 2:
                # Handle cases where only two tokens are returned
                surprisal_non_head = surprisal_values[0]
                surprisal_head = surprisal_values[1]
            else:
                print(f"Unexpected token count ({token_count}) for sentence: {sentence}")
                continue

            data.append([category_name, non_head, head, surprisal_non_head, surprisal_head])
            print(f"{sentence}: Non-Head: {surprisal_non_head}, Head: {surprisal_head}")  # Debugging print

# Process and print results
compound_combination("Regular Plural", regular_non_heads)
compound_combination("Irregular Plural", irregular_non_heads)
compound_combination("Pluralia Tantum", pluralia_tantum)

output_file = "results_gpt_j.csv"
df = pd.DataFrame(data, columns=["Category", "Non-Head", "Head", "Surprisal Non-head", "Surprisal head"])
df.to_csv(output_file, index=False)

print('\n results in results_gpt_j.csv \n')
