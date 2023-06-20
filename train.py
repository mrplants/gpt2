import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from AbstractDataset import AbstractDataset, collate_fn
from GPT2 import GPT2
from verify import generate_text
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_model(model, train_dataset, val_dataset, epochs=1, batch_size=64, learning_rate=0.001):
    """Trains the GPT-2 model on the given dataset.
    
    Args:
        model:  The GPT-2 model to train.
        dataset:  The dataset to train on.
        epochs:  The number of epochs to train for.
        batch_size:  The batch size to use.
        learning_rate:  The learning rate to use.
    """
    # Move the model to the cuda device if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create the DataLoader for batching the dataset
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Use Adam for optimization
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train() # Set the model to training mode

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        train_loss = 0
        for batch_idx, batch in enumerate(train_data_loader):
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1} of {len(train_data_loader)}")

            # Move batch to the device
            batch = batch.to(device)

            # Forward pass: get the model's predictions
            outputs = model(batch[:, :-1], mask=True)

            # Calculate loss:  Cross-entropy between predicted and actual next token.
            # Note: we flatten the outputs and targets to fit the loss function's input
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch[:, 1:].contiguous().view(-1))

            # Backward pass:  compute gradient of the loss with respect to model parameters
            loss.backward()
            train_loss += loss.item()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Perform the optimization step
            optimizer.step()

            # Zero the gradients to prevent them from accumulating
            optimizer.zero_grad()

            global_step = epoch * len(train_data_loader) + batch_idx

        # Calculate the validation loss
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0
            for batch in val_data_loader:
                batch = batch.to(device)
                outputs = model(batch[:, :-1], mask=True)
                val_ce_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch[:, 1:].contiguous().view(-1))
                val_loss += val_ce_loss.item()
        model.train() # Set the model back to training mode

        # Save the training and validation loss
        writer.add_scalar('loss/train', train_loss / len(train_data_loader), global_step)
        writer.add_scalar('loss/validation', val_loss / len(val_data_loader), global_step)

        # Save a generated sentence
        start_text = "The discovery of"
        gen = generate_text(model, start_text)
        writer.add_text('Generated Text', gen, global_step)

        print(f'Finished epoch {epoch+1} of {epochs}, train_loss={train_loss / len(train_data_loader)}, val_loss={val_loss / len(val_data_loader)}')
        print('Generated Text: ' + gen)

        # Save the model checkpoint
        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch}.pth")

writer.close()
# Create the dataset
dataset = AbstractDataset()

# Training parameters
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Model parameters
VOCAB_SIZE = len(dataset.tokenizer)
EMBED_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 1024

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create the model
model = GPT2(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH)

# If there are multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training via DataParallel.")
    # Use torch.nn.DataParallel to use multiple GPUs in parallel
    # This will take care of synchronizing the gradients between the replicas
    # model = torch.nn.DataParallel(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model):,} parameters.")

# Train the model
train_model(model, train_dataset, val_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE)
