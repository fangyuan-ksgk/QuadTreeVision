# Bernouli Sampling Selection Experiment
# Input Values are assgined Probability (in its logits value) by A Neural Network
# Selection is done by Bernouli Sampling based on selection probability
# STE & Gumbel Trick is used to encourage learnable selection

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the model
class SelectorModel(nn.Module):
    def __init__(self, input_dim, temperature=1.0, mode='gumbel'):
        super(SelectorModel, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.temperature = temperature
        self.mode = mode
    
    def forward(self, x):
        # Predict logits
        logits = self.fc(x)
        probs = torch.sigmoid(logits / self.temperature)
        
        if self.mode == 'ste':
            # Bernoulli sampling with STE
            binary_mask = torch.bernoulli(probs)
            scale_factor = 1.0
            mask = scale_factor * (probs - probs.detach()) + binary_mask
            
        elif self.mode == 'gumbel':
            # Gumbel Trick
            logits_cat = torch.stack([logits, -logits], dim=-1) # Negative logits to represent the other class
            gumbel_out = F.gumbel_softmax(logits_cat, tau=self.temperature, hard=True) 
            mask = gumbel_out[:, :, 0] # Selecting one of the two classes (0 or 1)
            
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose between 'ste' and 'gumbel'.")

        # Selective sum based on the mask
        selected_sum = torch.sum(mask * x, dim=1)
        return selected_sum, probs


# Hyperparameters
input_dim = 10
epochs = 20000
lr = 0.01

# SelectorModel remains unchanged

def train_and_evaluate_model(mode):
    model = SelectorModel(input_dim, temperature=1.0, mode=mode)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # Dummy dataset
    inputs = torch.randn(100, input_dim)
    # For simplicity, let's assume the GT is the sum of the first 5 values
    ground_truths = inputs[:, :5].sum(dim=1)
    reg_lambda = 0.2  # Regularization strength

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Model prediction
        predictions, probs = model(inputs)

        # Compute loss
        l1_loss = criterion(predictions, ground_truths)    
        regularization = reg_lambda * torch.mean(probs * (1 - probs))
        loss = l1_loss + regularization

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Mode: {mode}, Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    # Check the learned probabilities
    with torch.no_grad():
        _, probs = model(inputs)
        print(f"\nMode: {mode}, Final Probabilities: {probs.mean(dim=0)}")
        print("="*50)

print("Training model with Gumbel trick...")
train_and_evaluate_model('gumbel')

print("\nTraining model with STE...")
train_and_evaluate_model('ste')