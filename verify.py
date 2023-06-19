from transformers import GPT2Tokenizer
import torch
from GPT2 import GPT2
import os
from AbstractDataset import AbstractDataset, collate_fn

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text(model, start_text, max_length=50):
    model.eval() # Set the model to evaluation mode

    # Tokenize the starting text
    encoded_text = tokenizer.encode(start_text, return_tensors='pt').to(model.device)
    output_sequence = encoded_text # Initialize the generated text

    # Generate text
    with torch.no_grad(): # Disable gradient tracking to speed up the generation
        for _ in range(max_length):
            # Get the prediction for the next word
            predictions = model(output_sequence, None)
            predictions = predictions[:, -1:, :] # Get the last word's prediction

            # Sample from the predictions
            _, next_word = torch.max(predictions, dim=2) # Greedy decoding

            # Concatenate the generated word to the output sequence
            output_sequence = torch.cat([output_sequence, next_word], dim=1)
    
    # Decode the output sequence
    generated_sequence = tokenizer.decode(output_sequence[0])

    return generated_sequence

if __name__ == '__main__':

    # Create the dataset
    dataset = AbstractDataset()

    # Model parameters
    VOCAB_SIZE = len(dataset.tokenizer)
    EMBED_SIZE = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    FORWARD_EXPANSION = 4
    DROPOUT = 0.1
    MAX_LENGTH = 1024

    # Create the model
    model = GPT2(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FORWARD_EXPANSION, DROPOUT, MAX_LENGTH)

    for file in sorted(os.listdir('./checkpoints')):
        model.load_state_dict(torch.load('./checkpoints/'+file))
        model.eval()
        print(file)
        print(generate_text(model, "The discovery of"))
        print('\n')