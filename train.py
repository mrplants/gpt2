import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from GPT2 import GPT2
from verify import generate_text
from torch.utils.tensorboard import SummaryWriter
import datasets
from transformers import GPT2Tokenizer

# Training parameters
EPOCHS = 50
BATCH_SIZE = 3
LEARNING_RATE = 0.001

# Model parameters
EMBED_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 1024

def shift_chunk(example):
    chunks = []
    for i in range(len(example)):
        remaining = example[i:]
        chunks.append(remaining[:min(len(remaining), MAX_LENGTH)])
    return chunks

import torch

def create_causal_mask(token_mask):
    """
    Transforms a padding mask (1s for content, 0s for padding) into a causal mask,
    where 0s represent areas where the attention should look (content in its own and previous positions)
    and -inf represent areas where the attention should not look (content in future positions and padding areas).
    """
    # Create a tensor with ones on the diagonal and below (i.e., lower triangular matrix)
    causal_mask = torch.tril(torch.ones((token_mask.shape[1], token_mask.shape[1]))).to(token_mask.device)

    # Expand dimensions of padding mask to be [batch_size, sequence_length, sequence_length]
    padding_mask = token_mask.ne(tokenizer.pad_token_id).unsqueeze(1).expand(-1, token_mask.shape[1], -1)

    # Combine causal and padding masks, keeping smallest value at each position
    combined_mask = torch.min(padding_mask, causal_mask)

    # Set -inf to the positions where mask is 0, and 0 where mask is 1
    mask = combined_mask.masked_fill(combined_mask == 0, float('-inf')).masked_fill(combined_mask == 1, float(0))

    return mask.to(token_mask.device)

def train_model(model, train_dataset, val_dataset, tokenizer, epochs=1, batch_size=64, learning_rate=0.001):
    """Trains the GPT-2 model on the given dataset.
    
    Args:
        model:  The GPT-2 model to train.
        train_dataset:  The dataset to train on.
        val_dataset:  The dataset to validate on.
        tokenizer:  The GPT-2 tokenizer.
        epochs:  The number of epochs to train for.
        batch_size:  The batch size to use.
        learning_rate:  The learning rate to use.
    """
    # Move the model to the cuda device if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create the DataLoader for batching the dataset
    train_data_loader = DataLoader(train_dataset)
    val_data_loader = DataLoader(val_dataset)

    # Use Adam for optimization
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train() # Set the model to training mode
    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        cum_train_batches = 0
        cum_train_loss = 0
        for example in train_data_loader:
            chunked = shift_chunk(example['input_ids'])
            chunked_mask = shift_chunk(example['attention_mask'])
            for batch_idx in range(len(chunked)//batch_size):
                batch = chunked[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(chunked))]
                mask = chunked_mask[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(chunked_mask))]

                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx+1} of {len(chunked)//batch_size}")

                # Move batch to the device
                batch = torch.tensor(batch).to(device)
                mask = create_causal_mask(torch.tensor(mask)[:, :-1].bool()).repeat(NUM_HEADS, 1, 1).to(device)

                # Forward pass: get the model's predictions
                outputs = model(batch[:, :-1], mask=mask)

                # Calculate loss:  Cross-entropy between predicted and actual next token.
                # Note: we flatten the outputs and targets to fit the loss function's input
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch[:, 1:].contiguous().view(-1))

                # Backward pass:  compute gradient of the loss with respect to model parameters
                loss.backward()
                cum_train_loss += loss.item()
                cum_train_batches += 1

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Perform the optimization step
                optimizer.step()

                # Zero the gradients to prevent them from accumulating
                optimizer.zero_grad()


                if batch_idx % 1000 == 0 and batch_idx != 0:
                    print(f"Batch {batch_idx+1}")

                    # Calculate the validation loss
                    model.eval() # Set the model to evaluation mode
                    with torch.no_grad():
                        cum_val_batches = 0
                        cum_val_loss = 0
                        for example in val_data_loader:
                            chunked = shift_chunk(example['input_ids'])
                            chunked_mask = shift_chunk(example['attention_mask'])
                            for val_batch_idx in range(len(chunked)//batch_size):
                                print(f"Val Batch {val_batch_idx+1} of {len(chunked)//batch_size}")
                                batch = chunked[val_batch_idx*batch_size:min((val_batch_idx+1)*batch_size, len(chunked))]
                                mask = chunked_mask[val_batch_idx*batch_size:min((val_batch_idx+1)*batch_size, len(chunked_mask))]

                                # Move batch to the device
                                batch = torch.tensor(batch).to(device)
                                mask = create_causal_mask(torch.tensor(mask)[:, :-1].bool()).to(device).repeat(NUM_HEADS, 1, 1)

                                # Forward pass: get the model's predictions
                                outputs = model(batch[:, :-1], mask=mask)

                                # Calculate loss:  Cross-entropy between predicted and actual next token.
                                # Note: we flatten the outputs and targets to fit the loss function's input
                                val_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch[:, 1:].contiguous().view(-1))

                                # Backward pass:  compute gradient of the loss with respect to model parameters
                                cum_val_loss += val_loss.item()
                                cum_val_batches += 1

                    model.train() # Set the model back to training mode

                    # Save the training and validation loss
                    writer.add_scalar('loss/train', cum_train_loss / cum_train_batches, global_step)
                    cum_train_batches = 0
                    cum_train_loss = 0
                    writer.add_scalar('loss/validation', cum_val_loss / cum_val_batches, global_step)

                    # Save a generated sentence
                    start_text = "The discovery of"
                    gen = generate_text(model, start_text)
                    writer.add_text('Generated Text', gen, global_step)
                    print('Generated Text: ' + gen)

                global_step += 1

        print(f'Finished epoch {epoch+1} of {epochs}')

        # Save the model checkpoint
        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch}.pth")

    writer.close()


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)

def tokenize(example):
    return tokenizer(example["content"])

dataset = datasets.load_dataset('/home/grinch/ArXiv_dataset/ArXiv_dataset.py', split='default', streaming=True).map(tokenize)
# dataset = dataset.shuffle(seed=42, buffer_size=100)
val_dataset = dataset.take(1)
train_dataset = dataset.skip(1)

writer = SummaryWriter()

# Create the model
model = GPT2(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model):,} parameters.")

# Train the model
train_model(model, train_dataset, val_dataset, tokenizer, EPOCHS, BATCH_SIZE, LEARNING_RATE)
