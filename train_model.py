import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Networks import Action_Conditioned_FF
from Data_Loaders import Data_Loaders  


def train_model(no_epochs):
    # Parameters
    batch_size = 16
    learning_rate = 0.01
    
    # Initialize data loaders
    data_loaders = Data_Loaders(batch_size)
    
    # Initialize model, loss function, optimizer
    model = Action_Conditioned_FF(input_dim=6, hidden_dim=200, output_dim=1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training and evaluation losses
    train_losses = []
    eval_losses = []
    best_loss = float('inf')
    
    # Training loop
    for epoch_i in range(no_epochs):
        model.train()
        running_loss = 0.0

        for idx, sample in enumerate(data_loaders.train_loader):
            inputs, labels = sample['input'], sample['label']
            
            # Forward pass
            outputs = model(inputs)
            labels = labels.view(-1, 1)  # Reshape labels to match output shape
            loss = loss_function(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if idx % 50 == 0:
                print(f'Epoch [{epoch_i + 1}], Batch [{idx + 1}] - Loss: {loss.item():.4f}')
        
        # Calculate average train loss for this epoch
        avg_train_loss = running_loss / len(data_loaders.train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation on test set
        eval_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        eval_losses.append(eval_loss)
        
        print(f'Epoch [{epoch_i + 1}] - Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}')

        # Save best model based on evaluation loss
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), 'saved/saved_model.pkl')
    
    # Plot training and evaluation loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.show()
    
    return train_losses, eval_losses


if __name__ == '__main__':
    no_epochs = 500
    train_model(no_epochs)
