import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load the dataset
merged_df = pd.read_csv('/Users/rodneyfrazier/Desktop/flight/Merged_Flight_Data.csv')

# Define a function to classify problems related to engine
def is_engine_issue(problem_description):
    if isinstance(problem_description, str):
        if 'engine' in problem_description.lower():
            return 1
    return 0

# Recreate the 'engine_issue' column
merged_df['engine_issue'] = merged_df['PROBLEM'].fillna('').apply(is_engine_issue)

# Filter out rows where both 'PROBLEM' and 'ACTION' are empty
merged_df = merged_df[~((merged_df['PROBLEM'].isna()) & (merged_df['ACTION'].isna()))]

# Check for NaNs in the 'engine_issue' column and drop rows if necessary
merged_df = merged_df.dropna(subset=['engine_issue'])

# Prepare the text data
merged_df['text_data'] = merged_df['PROBLEM'].fillna('') + " " + merged_df['ACTION'].fillna('')

# Tokenize the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(list(merged_df['text_data']), max_length=512, truncation=True, padding=True, return_tensors='pt')

# Prepare the labels
labels = torch.tensor(merged_df['engine_issue'].values)

# Ensure token and label counts match
print(f"Number of tokens: {tokens['input_ids'].size(0)}")
print(f"Number of labels: {labels.size(0)}")

# Continue with the rest of the code only if the sizes match

# Split data into training and testing sets
train_tokens, test_tokens = tokens[:int(len(tokens)*0.8)], tokens[int(len(tokens)*0.8):]
train_labels, test_labels = labels[:int(len(labels)*0.8)], labels[int(len(labels)*0.8):]

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    evaluation_strategy="epoch",     
    save_total_limit=1,             
)

# Create Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=torch.utils.data.TensorDataset(train_tokens['input_ids'], train_labels),
    eval_dataset=torch.utils.data.TensorDataset(test_tokens['input_ids'], test_labels),
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate()

print(results)
