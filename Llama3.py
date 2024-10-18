import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd

# Load LLaMA model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")

# Load dataset
merged_df = pd.read_csv('/Users/rodneyfrazier/Desktop/flight/Merged_Flight_Data.csv')

# Create 'engine_issue' classification based on the problem description
def is_engine_issue(problem_description):
    if isinstance(problem_description, str):
        if 'engine' in problem_description.lower():
            return 1
    return 0

merged_df['engine_issue'] = merged_df['PROBLEM'].fillna('').apply(is_engine_issue)

# Prepare the text data
merged_df['text_data'] = merged_df['PROBLEM'].fillna('') + " " + merged_df['ACTION'].fillna('')

# Function to classify using LLaMA
def classify_with_llama(text):
    prompt = f"Based on the following maintenance issue description, classify if it involves an engine issue: \n\n{text}\n\nReply with either '1' for engine issue or '0' for no engine issue."
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=100)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract classification result from the response
    classification = response.strip().split()[-1]
    return int(classification) if classification in ['0', '1'] else None

# Apply LLaMA 3 classification to the dataset
merged_df['llama3_classification'] = merged_df['text_data'].apply(classify_with_llama)

# Evaluate the performance
correct_predictions = (merged_df['llama3_classification'] == merged_df['engine_issue']).sum()
total_predictions = merged_df['llama3_classification'].notna().sum()

accuracy = correct_predictions / total_predictions
print(f"LLaMA 3 classification accuracy: {accuracy:.2f}")
