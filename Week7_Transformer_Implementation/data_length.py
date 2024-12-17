import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

dataset = load_dataset("bentrevett/multi30k", split="train")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

source_lengths = []
target_lengths = []

for example in dataset:
    source_text = example["en"]
    target_text = example["de"]

    # Tokenize and measure length
    source_length = len(tokenizer(source_text)["input_ids"])
    target_length = len(tokenizer(target_text)["input_ids"])

    source_lengths.append(source_length)
    target_lengths.append(target_length)

# Calculate average lengths
avg_source_length = np.mean(source_lengths)
avg_target_length = np.mean(target_lengths)

# Display results
print(f"Average Source Length: {avg_source_length}")
print(f"Average Target Length: {avg_target_length}")

plt.figure(figsize=(10, 6))
plt.hist(source_lengths, bins=range(0, max(source_lengths)+1, 5), alpha=0.5, color='blue', label="Source Lengths")
plt.hist(target_lengths, bins=range(0, max(target_lengths)+1, 5), alpha=0.5, color='purple', label="Target Lengths")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
