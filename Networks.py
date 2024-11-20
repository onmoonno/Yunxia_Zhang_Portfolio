import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=200, output_dim=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        # Initialize nn.Module
        super(Action_Conditioned_FF, self).__init__()
        
        # Define the layers
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        
        # Define activation functions
        self.activation_hidden = nn.ReLU()
        self.activation_output = nn.Sigmoid()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
         # Pass input through the network layers
        hidden = self.activation_hidden(self.input_to_hidden(input))
        output = self.activation_output(self.hidden_to_output(hidden))
        
        return output

    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
         # Evaluate the model on the test data
        loss_sum = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient tracking for evaluation
            for idx, sample in enumerate(test_loader):
                inputs, labels = sample['input'], sample['label']
                outputs = model(inputs)
                loss_sum += loss_function(outputs, labels.view(-1, 1)).item()
        
        # Calculate average loss
        loss = loss_sum / len(test_loader)
        print(f'Test Loss: {loss:.3f}')
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
