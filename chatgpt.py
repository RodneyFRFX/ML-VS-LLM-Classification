import openai
import pandas as pd

# Load your dataset
merged_df = pd.read_csv('/Users/rodneyfrazier/Desktop/flight/Merged_Flight_Data.csv')

# Define a function to classify problems related to engine
def is_engine_issue(problem_description):
    if isinstance(problem_description, str):
        if 'engine' in problem_description.lower():
            return 1
    return 0

# Recreate the 'engine_issue' column
merged_df['engine_issue'] = merged_df['PROBLEM'].fillna('').apply(is_engine_issue)

# Combine text data
merged_df['text_data'] = merged_df['PROBLEM'].fillna('') + " " + merged_df['ACTION'].fillna('')

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Define a function to classify using GPT-4
def classify_with_gpt4(text):
    prompt = f"Based on the following maintenance issue description, classify if it involves an engine issue: \n\n{text}\n\nReply with either '1' for engine issue or '0' for no engine issue."
    
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )
    
    classification = response['choices'][0]['text'].strip()
    
    # Return 1 if engine issue, 0 if no engine issue
    return int(classification) if classification in ['0', '1'] else None

# Apply GPT-4 classification to your dataset
merged_df['gpt4_classification'] = merged_df['text_data'].apply(classify_with_gpt4)

# Compare GPT-4's classifications with the true labels
correct_predictions = (merged_df['gpt4_classification'] == merged_df['engine_issue']).sum()
total_predictions = merged_df['gpt4_classification'].notna().sum()

accuracy = correct_predictions / total_predictions
print(f"GPT-4 classification accuracy: {accuracy:.2f}")
