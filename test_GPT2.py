import torch
from torch import nn
import unittest
from GPT2 import GPT2

class TestGPT2(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.embed_size = 128
        self.num_layers = 6
        self.num_heads = 8
        self.forward_expansion = 4
        self.dropout = 0.1
        self.max_length = 100
        self.model = GPT2(
            self.vocab_size,
            self.embed_size,
            self.num_layers,
            self.num_heads,
            self.forward_expansion,
            self.dropout,
            self.max_length)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def test_forward(self):
        # Create a random input tensor with shape (batch_size, seq_length)
        x = torch.randint(0, self.vocab_size, (1, self.max_length))
        # Create a random attention mask
        mask = torch.zeros(self.max_length, self.max_length).bool()

        # Run the forward pass
        output = self.model.forward(x, mask)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (1, self.max_length, self.vocab_size))
    
    def test_train(self):
        # Create a random input tensor with shape (batch_size, seq_length)
        x = torch.randint(0, self.vocab_size, (1, self.max_length)).to(self.model.device)

        # Create a random attention mask
        mask = torch.zeros(self.max_length, self.max_length).bool().to(self.model.device)

        # Also create a random target tensor
        targets = torch.randint(0, self.vocab_size, (1, self.max_length)).to(self.model.device)

        # Run the forward pass
        output = self.model.forward(x, mask)

        # Compute the loss
        loss = self.loss_function(output.view(-1, self.vocab_size), targets.view(-1))

        # Zero the gradients
        self.optimizer.zero_grad()

        # Perform the backward pass
        loss.backward()

        # Update the weights
        self.optimizer.step()

        # Check that the loss is decreasing after one training step
        new_output = self.model.forward(x, mask)
        new_loss = self.loss_function(new_output.view(-1, self.vocab_size), targets.view(-1))
        self.assertLess(new_loss.item(), loss.item())

if __name__ == "__main__":
    unittest.main()