import torch

# Load the original checkpoint
checkpoint = torch.load("gpt2-text-classifier-model.pt", map_location="cpu")

# Adjust the dimensions of fc1's weight and bias for 2-class classification
checkpoint["fc1.weight"] = checkpoint["fc1.weight"][:2, :]
checkpoint["fc1.bias"] = checkpoint["fc1.bias"][:2]

# Save the updated checkpoint
torch.save(checkpoint, "gpt2-text-classifier-model-2class.pt")

print("Checkpoint adjusted for 2 classes and saved as 'gpt2-text-classifier-model-2class.pt'")
