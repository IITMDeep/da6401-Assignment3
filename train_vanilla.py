# Import Lib

import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import random
import heapq
import wandb
import argparse

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#wandb login 6a66920f640c7001ec17ad4aa7a5da8b378aee61

"""# Preprocessing"""

def encode(x, max_length, char_to_idx):
    """
    Encode a string into a tensor.

    Args:
    - x (str): Input string to encode.
    - max_length (int): Maximum length for the encoded tensor.
    - char_to_idx (dict): Character to index mapping.

    Returns:
    - encoded (torch.Tensor): Encoded tensor.
    - length (int): Actual length of the encoded sequence.
    """
    encoded = np.zeros(max_length, dtype=int)
    encoder = np.array([char_to_idx[char] for char in x])
    length = min(max_length, len(encoder))
    encoded[:length] = encoder[:length]

    return torch.tensor(encoded, dtype=torch.int64), length

def get_tensor_object(df, max_input_length, max_output_length, char_to_idx_input, char_to_idx_output):
    """
    Create tensor objects from a DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame containing input and output sequences.
    - max_input_length (int): Maximum length for input sequences.
    - max_output_length (int): Maximum length for output sequences.
    - char_to_idx_input (dict): Character to index mapping for input sequences.
    - char_to_idx_output (dict): Character to index mapping for output sequences.

    Returns:
    - tensor_inputs (torch.Tensor): Tensor containing encoded input sequences.
    - tensor_outputs (torch.Tensor): Tensor containing encoded output sequences.
    """
    
    # Encode unique inputs and outputs into tensors
    encoded_inputs = []
    encoded_outputs = []

    # Encode the input column
    for input_str in df[0]:
        encoded_input, input_length = encode(input_str, max_input_length, char_to_idx_input)
        encoded_inputs.append(encoded_input)

    # Encode the output column
    for output_str in df[1]:
        encoded_output, output_length = encode(output_str, max_output_length, char_to_idx_output)
        encoded_outputs.append(encoded_output)

    # Stack tensors column-wise
    
#     tensor_inputs = torch.stack(encoded_inputs, dim=1)
#     tensor_outputs = torch.stack(encoded_outputs, dim=1)
    tensor_inputs = torch.stack(encoded_inputs)
    tensor_outputs = torch.stack(encoded_outputs)

    return tensor_inputs, tensor_outputs

def load_dataset(path):
    """
    Load a dataset from a TSV file.
    Args:
    - path (str): Path to the TSV file.
    Returns:
    - df (pd.DataFrame): Loaded DataFrame.
    - max_input_length (int): Maximum length for input sequences.
    - max_output_length (int): Maximum length for output sequences.
    """
    df = pd.read_csv(path, header=None, encoding='utf-8', sep='\t')  # Changed separator to tab
    
    # Convert values to strings before adding special characters
    df[0] = df[0].astype(str).apply(lambda x: x + '$')
    df[1] = df[1].astype(str).apply(lambda x: '^' + x + '$')
    
    # Determine maximum length for input and output sequences
    max_input_length = max(len(x) for x in df[0].unique())
    max_output_length = max(len(x) for x in df[1].unique())
    return df, max_input_length, max_output_length

def look_up_table(vocab1, vocab2, vocab3):
    """
    Create lookup tables for vocabulary mapping.

    Args:
    - vocab1 (list): First list of vocabulary items.
    - vocab2 (list): Second list of vocabulary items.
    - vocab3 (list): Third list of vocabulary items.

    Returns:
    - vocab_to_int (dict): Mapping from vocabulary items to integers.
    - int_to_vocab (dict): Mapping from integers to vocabulary items.
    """
    
    # Combine all vocabularies into one set
    vocab = set(''.join(vocab1) + ''.join(vocab2) + ''.join(vocab3))
    vocab.discard('^')  
    vocab.discard('$')  
    vocab_to_int = {"": 0, '^':1, '$':2}
    for v_i, v in enumerate(sorted(vocab), len(vocab_to_int)):
        vocab_to_int[v] = v_i
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab



"""# Create Seq2Seq Model

## encoder and decoder
"""

class Encoder(nn.Module): 
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, cell_type):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        
        # Define embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Define RNN layer with specific cell type
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', or 'RNN'.")
        
        
    def forward(self, x): # x shape: (seq_length, N) where N is batch size
        # Perform dropout on the input
        embedding = self.embedding(x)
        embedding = self.dropout(embedding) # embedding shape: (seq_length, N, embedding_size)
        
        if self.cell_type == "LSTM":
            # Pass through the LSTM layer
            outputs, (hidden, cell) = self.rnn(embedding) # outputs shape: (seq_length, N, hidden_size)
            if self.bidirectional:
                # Sum the bidirectional outputs
                outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
                hidden = torch.cat((hidden[: self.num_layers], hidden[self.num_layers:]), dim=0)
            # Return hidden state and cell state   
            return hidden, cell
        elif self.cell_type == "GRU" or self.cell_type == "RNN":
            # Pass through the RNN/GRU layer
            outputs, hidden = self.rnn(embedding) # outputs shape: (seq_length, N, hidden_size)
            if self.bidirectional:
                # Sum the bidirectional outputs
                outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
                hidden = torch.cat((hidden[: self.num_layers], hidden[self.num_layers:]), dim=0)

            # Return hidden state and cell state
            return hidden
        else:
            print("Invalid cell_type specified for Encoder.")
            return None


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout, bidirectional, cell_type):
        super(Decoder, self).__init__()
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)  
        self.num_layers = num_layers 
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        
        # Define embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Define RNN layer with specific cell type
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', or 'RNN'.")
            
            
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)  # Adjust input size for bidirectional decoder
        # Softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden, cell): # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        
        # Ensure x has the shape (1, N)
        x = x.unsqueeze(0)
        
        # Perform dropout on the input
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)  # embedding shape: (1, N, embedding_size)
        
        if self.cell_type == "LSTM":
            # Pass through the LSTM layer
            outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))  # outputs shape: (1, N, hidden_size * num_directions)

            # Pass through fully connected layer
            out = self.fc(outputs).squeeze(0)
            predictions = self.log_softmax(out)

            return predictions, hidden, cell
        elif self.cell_type == "GRU" or self.cell_type == "RNN":
            # Pass through the RNN/GRU layer
            outputs, hidden = self.rnn(embedding, hidden)  # outputs shape: (1, N, hidden_size * num_directions)

            # Pass through fully connected layer
            out = self.fc(outputs).squeeze(0)
            predictions = self.log_softmax(out)

            return predictions, hidden

        else:
            print("Invalid cell_type specified for Decoder.")
            return None


"""## Seq2Seq Class"""

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, output_char_to_int, teacher_forcing, cell_type):

        super(Seq2Seq, self).__init__()  
        # Initialize encoder and decoder
        self.decoder = decoder
        self.encoder = encoder
        self.cell_type = cell_type
        self.target_vocab_size = len(output_char_to_int)
        self.teacher_force_ratio = teacher_forcing
        
    def forward(self, source, target):
        # Get batch size, target length, and target vocabulary size
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.target_vocab_size
        teacher_force_ratio = self.teacher_force_ratio
        
        # Initialize outputs tensor
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(source.device)
        # Grab the first input to the Decoder which will be <SOS> token i.e '^'
        x = target[0]
        # Get hidden state and cell state from encoder
        if self.cell_type == 'LSTM':
            hidden, cell = self.encoder(source)
        else:
            hidden = self.encoder(source)
        
        for t in range(1, target_len):
            # Use previous hidden and cell states as context from encoder at start
            if self.cell_type == 'LSTM':
                output, hidden, cell = self.decoder(x, hidden, cell)
            else:
                output, hidden = self.decoder(x, hidden, None)
                
            # Store next output prediction
            outputs[t] = output
            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            # Update input for next time step based on teacher forcing ratio
            x = best_guess if random.random() >= teacher_force_ratio else target[t]

        return outputs

"""# TRAINING"""

# BEAM SEARCH FUNCTION
def beam_search(model, input_seq, max_length, input_char_index, output_char_index, reverse_target_char_index, beam_width, length_penalty, cell_type):
    """
    Perform beam search to generate a sequence using the provided model.

    Args:
    - model (nn.Module): The Seq2Seq model.
    - input_seq (str): The input sequence.
    - max_length (int): Maximum length of the input sequence.
    - input_char_index (dict): Mapping from characters to integers for the input vocabulary.
    - output_char_index (dict): Mapping from characters to integers for the output vocabulary.
    - reverse_target_char_index (dict): Reverse mapping from integers to characters for the output vocabulary.
    - beam_width (int): Beam width for beam search.
    - length_penalty (float): Length penalty for beam search.
    - cell_type (str): Type of RNN cell used in the model ('LSTM', 'GRU', or 'RNN').

    Returns:
    - str: The generated output sequence.
    """
    if len(input_seq) > max_length:
        print("Input Length is exceeding max length!!!!")
        return ""

    # Create np array of zeros of length input
    input_data = np.zeros((max_length, 1), dtype=int)  # (N,1)

    # Encode the input
    for idx, char in enumerate(input_seq):
        input_data[idx, 0] = input_char_index[char]
    input_data[idx + 1, 0] = input_char_index["$"]  # EOS

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.int64).to(device)  # N,1

    with torch.no_grad():
        if cell_type == 'LSTM':
            hidden, cell = model.encoder(input_tensor)

        else:
            hidden = model.encoder(input_tensor)

    # Initialize beam
    out_t = output_char_index['^']
    out_reshape = np.array(out_t).reshape(1,)
    hidden_par = hidden.unsqueeze(0)
    initial_sequence = torch.tensor(out_reshape).to(device)
    beam = [(0.0, initial_sequence, hidden_par)]  # [(score, sequence, hidden)]

    for _ in range(len(output_char_index)):
        candidates = []
        for score, seq, hidden in beam:
            if seq[-1].item() == output_char_index['$']:
                # If the sequence ends with the end token, add it to the candidates
                candidates.append((score, seq, hidden))
                continue

            last_token = np.array(seq[-1].item()).reshape(1,)
            x = torch.tensor(last_token).to(device)

            if cell_type == 'LSTM':
                output, hidden, cell,  = model.decoder(x, hidden.squeeze(0), cell)
            else:
                output, hidden,  = model.decoder(x, hidden.squeeze(0), None)

            probabilities = F.softmax(output, dim=1)
            topk_probs, topk_tokens = torch.topk(probabilities, k=beam_width)

            for prob, token in zip(topk_probs[0], topk_tokens[0]):
                new_seq = torch.cat((seq, token.unsqueeze(0)), dim=0)
                seq_length_norm_factor = (len(new_seq) - 1) / 5
                candidate_score = score + torch.log(prob).item() / (seq_length_norm_factor ** length_penalty)
                candidates.append((candidate_score, new_seq, hidden.unsqueeze(0)))

        # Select top-k candidates based on the accumulated scores
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

    best_score, best_sequence, _ = max(beam, key=lambda x: x[0])  # Select the best sequence from the beam as the output

    # Convert the best sequence indices to characters
    return ''.join([reverse_target_char_index[token.item()] for token in best_sequence[1:]])


# TRAINING FUNCTION
def train(model, num_epochs, criterion, optimizer, train_batch_x, train_batch_y, val_batch_x, val_batch_y, df_val, input_char_to_int, output_char_to_int, output_int_to_char, beam_width, length_penalty, cell_type, max_length, wandb_log):
    """
    Train the Seq2Seq model.

    Args:
    - model (nn.Module): The Seq2Seq model.
    - num_epochs (int): Number of training epochs.
    - criterion: Loss criterion for training.
    - optimizer: Optimizer for training.
    - train_batch_x: Training input data.
    - train_batch_y: Training target data.
    - val_batch_x: Validation input data.
    - val_batch_y: Validation target data.
    - df_val: DataFrame for validation data.
    - input_char_to_int (dict): Mapping from characters to integers for the input vocabulary.
    - output_char_to_int (dict): Mapping from characters to integers for the output vocabulary.
    - output_int_to_char (dict): Reverse mapping from integers to characters for the output vocabulary.
    - beam_width (int): Beam width for beam search.
    - length_penalty (float): Length penalty for beam search.
    - cell_type (str): Type of RNN cell used in the model ('LSTM', 'GRU', or 'RNN').
    - max_length (int): Maximum length of sequences.
    - wandb_log (int): Whether to log to wandb (1 or 0).
    Returns:
    - nn.Module: The trained model.
    - float: Validation accuracy.
    """
    for epoch in range(num_epochs):
        total_words = 0
        correct_pred = 0
        total_loss = 0
        accuracy = 0
        model.train()
        
        # Use tqdm for progress tracking
        train_data_iterator = tqdm(zip(train_batch_x, train_batch_y), total=len(train_batch_x))
        
        for (x, y) in train_data_iterator:
            # Get input and targets and move to device
            target, inp_data = y.to(device), x.to(device)
            
            # Forward propagation
            optimizer.zero_grad()
            output = model(inp_data, target)
            
            target = target.reshape(-1)
            output = output.reshape(-1, output.shape[2])
            
            pad_mask = (target != 0)  
            target = target[pad_mask] # Select non-padding elements
            output = output[pad_mask] 
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backpropagation
            loss.backward()
            
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate total loss
            total_loss += loss.item()
            # Update total words processed
            total_words += target.size(0)
            # Calculate number of correct predictions
            correct_pred += torch.sum(torch.argmax(output, dim=1) == target).item()
            
        # Calculate average loss per batch
        avg_loss = total_loss / len(train_batch_x)
        # Calculate accuracy
        accuracy = 100*correct_pred / total_words
        
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            val_total_words = 0
            val_correct_pred = 0

            val_data_iterator = tqdm(zip(val_batch_x, val_batch_y), total=len(val_batch_x))
            for x_val, y_val in val_data_iterator:
                target_val, inp_data_val = y_val.to(device), x_val.to(device)
                output_val = model(inp_data_val, target_val)
                
                
                target_val = target_val.reshape(-1)
                output_val = output_val.reshape(-1, output_val.shape[2])
                
                pad_mask = (target_val != 0)  
                target_val = target_val[pad_mask] # Select non-padding elements
                output_val = output_val[pad_mask] 
            
                val_loss = criterion(output_val, target_val)
                val_total_loss += val_loss.item()
                val_total_words += target_val.size(0)
                val_correct_pred += torch.sum(torch.argmax(output_val, dim=1) == target_val).item()

            # Calculate validation statistics
            val_accuracy = 100*val_correct_pred / val_total_words
            val_avg_loss = val_total_loss / len(val_batch_x)

            
            
        # Total word predict correct over training
        beam_val_pred = 0
        beam_val = 0
        for i in tqdm(range(df_val.shape[0])):
            input_seq = df_val.iloc[i, 0][:-1] 
            true_seq = df_val.iloc[i, 1][1:-1]
            predicted_output = beam_search(model, input_seq, max_length, input_char_to_int, output_char_to_int, output_int_to_char, beam_width, length_penalty, cell_type)
            if true_seq == predicted_output[:-1]:
                beam_val_pred+=1
        beam_val = 100*beam_val_pred/df_val.shape[0]



        # Print statistics
        print(f"Epoch {epoch + 1} / {num_epochs} ===========================>")
        print(f"Train Accuracy Char: {accuracy:.4f}, Train Average Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy Char: {val_accuracy:.4f}, Validation Average Loss: {val_avg_loss:.4f}")
        print(f"Beam Val Word Accuracy: {beam_val:.4f} Correct Prediction : {beam_val_pred}/{df_val.shape[0]}")    
        
        if wandb_log == 1:
            wandb.log({
                "train_accuracy_char": accuracy,
                "train_loss": avg_loss,
                "val_accuracy_char": val_accuracy,
                "val_loss": val_avg_loss,
                "beam_val_accuracy_word" : beam_val,
            })
        
    
    return model, beam_val

"""# Prediction"""
def predict(model, input_seq, max_length, input_char_index, output_char_index, reverse_target_char_index):
    """
    Generate text output using a trained model.

    Args:
        model (torch.nn.Module): The trained sequence-to-sequence model.
        input_seq (str): The input sequence to be translated.
        max_length (int): Maximum length of the input sequence.
        input_char_index (dict): A dictionary mapping input characters to their integer indices.
        output_char_index (dict): A dictionary mapping output characters to their integer indices.
        reverse_target_char_index (dict): A dictionary mapping integer indices to output characters.

    Returns:
        output_text (str): The generated output text.
        attentions (torch.Tensor): Attention scores during decoding.
    """

    model.eval()
    if len(input_seq) > max_length:
        print("Input Length is exceeding max length!!!!")
        return ""

    # Create np array of zero of length i/p
    input_data = np.zeros((max_length, 1), dtype=int) # (N,1)

    # Encode the input
    for idx, char in enumerate(input_seq):
        input_data[idx, 0] = input_char_index[char]
    input_data[idx+1, 0] = input_char_index["$"]

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.int64).to(device) # N,1

    with torch.no_grad():
        if cell_type == "LSTM":
            encoder_outputs, hidden_state, cell = model.encoder(input_tensor)
            hidden_state =  hidden_state[:model.decoder.num_layers]
            cell =  cell[:model.decoder.num_layers]

        else:
            encoder_outputs, hidden_state = model.encoder(input_tensor)
            hidden_state =  hidden_state[:model.decoder.num_layers]

    output_text = []
    output_start_token = output_char_index['^'] # SOS token
    output_start_token_tensor = torch.tensor([output_start_token]).to(device)

    attentions = torch.zeros(input_max_len + 1, 1, input_max_len + 1)
    #decoder_attentions = torch.zeros(29, 29)
    for i in range(1, len(output_char_index)):
        if cell_type == "LSTM":
            output, hidden_state, cell, attention = model.decoder(output_start_token_tensor, encoder_outputs, hidden_state, cell)
        else:
            output, hidden_state, attention = model.decoder(output_start_token_tensor, encoder_outputs, hidden_state, None)

        #print(attention)
        predicted_char = reverse_target_char_index[output.argmax(1).item()]
        attentions[i] = attention
        #decoder_attentions[i] = attention.data
        if predicted_char != '$':
            output_text.append(predicted_char)
        else:
            break
        output_start_token_tensor = torch.tensor([output.argmax(1)]).to(device)


    return ''.join(output_text), attentions[:i + 1]


def store_results(data_type, words, translations, predictions, results):
    """
    This function saves the evaluation results to a CSV file.

    Args:
        data_type (str): The type of data used for evaluation (e.g., 'val', 'test').
        words (list): List of source words (without start/end tokens).
        translations (list): List of reference translations (without start/end tokens).
        predictions (list): List of predicted translated sequences (without start/end tokens).
        results (list): List of 'Yes' or 'No' indicating correct/incorrect predictions.
    """

    # Create a dictionary to store the results in a structured format
    log = {
        'Word': words,
        'Translation': translations,
        'Prediction': predictions,
        'Result': results  # 'Yes' for correct, 'No' for incorrect
    }

    # Get the current directory path
    current_dir = os.getcwd()

    # Construct the file path for the CSV file in the current directory
    file_path = os.path.join(current_dir, f'prediction.csv')

    # Create a Pandas DataFrame from the dictionary
    data_frame = pd.DataFrame(log)

    # Save the DataFrame to a CSV file (header=True includes column names, index=False excludes row index)
    data_frame.to_csv(file_path, header=True, index=False)

"""## Main Function"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='kaggle/input/hinid-dataset/dakshina_sampled/hi', help='Path to the data folder')
    parser.add_argument('-l', '--lang', type=str, default='hi', help='Language for which training is to be done')
    parser.add_argument('-es', '--embedding_size', type=int, default=256, help='Embedding size')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('-nl', '--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('-ct', '--cell_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Cell type (RNN, LSTM, GRU)')
    parser.add_argument('-dr', '--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-op', '--optimizer', type=str, default='adagrad', choices=['adam', 'sgd', 'rmsprop', 'nadam', 'adagrad'], help='Optimizer (adam, sgd, rmsprop, nadam, adagrad)')
    parser.add_argument('-bw', '--beam_search_width', type=int, default=1, help='Beam search width')
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.6, help='Length penalty')
    parser.add_argument('-tf', '--teacher_forcing', type=float, default=0.7, help='Teacher forcing ratio')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='Use bidirectional encoder', choices=[True, False])
    parser.add_argument('-test', '--test_accuracy', help='Flag to enable test accuracy', type=int, default=0, choices=[0, 1])
    parser.add_argument('-store', '--store_csv', type=bool, default=False, help='Store Prediction CSV in curr dir', choices=[True, False])
    parser.add_argument('-wl', '--wandb_log', type=int, default=0, help='Whether to log to WandB (1 for yes, 0 for no)')
    parser.add_argument('-wp', '--wandb_project',help='Project name used to track experiments in Weights & Biases dashboard', type=str, default='DL_Assignment_3')
    parser.add_argument('-we', '--wandb_entity', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.', type=str, default='cs24m019-iitm')

    config = parser.parse_args()
    data_path = config.data_path
    lang = config.lang


    # Load Dataset
    df_train, train_input_len, train_out_len = load_dataset(f'/{data_path}/{lang}/{lang}_train.tsv')
    df_val, val_input_len, val_out_len = load_dataset(f'/{data_path}/{lang}/{lang}_dev.tsv')
    df_test, test_input_len, test_out_len = load_dataset(f'/{data_path}/{lang}/{lang}_test.tsv')

    input_max_len = max(train_input_len, val_input_len, test_input_len)
    output_max_len = max(train_out_len, val_out_len, test_out_len)

    max_length = max(input_max_len, output_max_len)

    # Create Look Up Table
    input_char_to_int, input_int_to_char = look_up_table(df_train[0], df_val[0], df_test[0])
    output_char_to_int, output_int_to_char = look_up_table(df_train[1], df_val[1], df_test[1])

    # Data Embedding and Converting them into Tensor
    train_inputs, train_outputs = get_tensor_object(df_train, max_length, max_length, input_char_to_int, output_char_to_int)
    val_inputs, val_outputs = get_tensor_object(df_val, max_length, max_length, input_char_to_int, output_char_to_int)
    test_inputs, test_outputs = get_tensor_object(df_test, max_length, max_length, input_char_to_int, output_char_to_int)

    # Add extra len to be on safe size and new inputs always under max length size
    max_length = max_length + 1

    # Transpose column wise
    train_inputs, train_outputs = torch.transpose(train_inputs, 0, 1), torch.transpose(train_outputs, 0, 1)
    val_inputs, val_outputs = torch.transpose(val_inputs, 0, 1), torch.transpose(val_outputs, 0, 1)
    test_inputs, test_outputs = torch.transpose(test_inputs, 0, 1), torch.transpose(test_outputs, 0, 1)

    # Initialize Hyperparameters
    input_size = len(input_char_to_int)
    output_size = len(output_char_to_int)
    embedding_size = config.embedding_size
    hidden_size = config.hidden_size
    enc_num_layers = config.num_layers
    dec_num_layers = config.num_layers
    cell_type = config.cell_type
    dropout = config.dropout
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    optimizer = config.optimizer
    beam_width = config.beam_search_width
    bidirectional = config.bidirectional
    length_penalty = config.length_penalty
    teacher_forcing = config.teacher_forcing

    # Create train data batch
    train_batch_x, train_batch_y = torch.split(train_inputs, batch_size, dim=1), torch.split(train_outputs, batch_size, dim=1)
    # Validation data batch
    val_batch_x, val_batch_y = torch.split(val_inputs, batch_size, dim=1), torch.split(val_outputs, batch_size, dim=1)


    # Intialize encoder, decoder and seq2seq model
    encoder = Encoder(input_size, embedding_size, hidden_size, enc_num_layers, dropout, bidirectional, cell_type).to(device)
    decoder = Decoder(output_size, embedding_size, hidden_size, output_size, dec_num_layers, dropout, bidirectional, cell_type).to(device)
    model = Seq2Seq(encoder, decoder, output_char_to_int, teacher_forcing, cell_type).to(device)

    # Print total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Total Trainable Parameters: {total_params}')


    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer == 'nadam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    # TRAINING

    if config.wandb_log == 1:
        run_config = vars(config).copy()
        run_config.pop('data_path', None)
        run_config.pop('lang', None)
        run_config.pop('wandb_log', None)
        run_config.pop('wandb_project', None)
        run_config.pop('wandb_entity', None)
        wandb.init(config=run_config, project=config.wandb_project, name=config.wandb_entity)
        wandb.run.name = 'cell_' + config.cell_type + '_bs_' + str(config.batch_size) + '_ep_' + str(config.num_epochs) + '_op_' + str(config.optimizer) + '_drop_' + str(config.dropout) + '_bsw_' + str(config.beam_search_width) +'_emb_' + str(config.embedding_size) + '_hs_' + str(config.hidden_size) + '_elayer_' + str(config.num_layers) + '_dlayer_' + str(config.num_layers)

    model, acc = train(model, num_epochs, criterion, optimizer, train_batch_x, train_batch_y, val_batch_x, val_batch_y, df_val, input_char_to_int, output_char_to_int, output_int_to_char, beam_width, length_penalty, cell_type, max_length, config.wandb_log)
    if config.wandb_log == 1:
        wandb.log({
                "accuracy": acc,
            })

    # PRINTING TEST ACCURACY
    if config.test_accuracy == 1:

        test_acc = 0
        correct_pred = 0
        words_test = []
        translations_test = []
        predictions_test = []
        results_test = []

        for i in tqdm(range(df_test.shape[0])):
            input_seq = df_test.iloc[i, 0][:-1]
            true_seq = df_test.iloc[i, 1][1:-1]
            predicted_output = beam_search(model, input_seq, max_length, input_char_to_int, output_char_to_int,
                                           output_int_to_char, beam_width, length_penalty, cell_type)
            words_test.append(input_seq)
            translations_test.append(true_seq)
            predictions_test.append(predicted_output[:-1])
            if true_seq == predicted_output[:-1]:
                correct_pred += 1
                results_test.append('Yes')
            else:
                results_test.append('No')

        test_acc = 100 * correct_pred / df_test.shape[0]

        print(f'Test Accuracy Word Level: {test_acc}, Correctly Predicted: {correct_pred}')

        if config.store_csv == True:
            store_results('test', words_test, translations_test, predictions_test, results_test)




