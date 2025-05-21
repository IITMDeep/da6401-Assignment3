# DA6401 - Assignment 3

---

# RNN BASED Seq2Seq MODEL

This repository contains a Python implementation of a sequence-to-sequence (Seq2Seq) model for sequence prediction tasks. The Seq2Seq model is implemented using PyTorch and includes different recurrent neural network (RNN) cell types such as LSTM, RNN, and GRU for both the encoder and decoder.


## Classes

### Encoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the encoder.
  - `forward()`: Performs forward pass through the encoder.

### Decoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `output_size`: Size of the output vocabulary.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the decoder.
  - `forward()`: Performs forward pass through the decoder.

### Seq2Seq
- **Data Members**:
  - `encoder`: Instance of the Encoder class.
  - `decoder`: Instance of the Decoder class.
  - `target_vocab_size`: Size of the target vocabulary.
  - `teacher_force_ratio`: Ratio of teacher forcing during training.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Methods**:
  - `__init__()`: Initializes the Seq2Seq model.
  - `forward()`: Performs forward pass through the model.

## Training Functions

### train()
- **Arguments**:
  - `model`: The Seq2Seq model to be trained.
  - `num_epochs`: Number of training epochs.
  - `criterion`: Loss criterion for training.
  - `optimizer`: Optimizer for training.
  - `train_batch_x`: Training input data batch.
  - `train_batch_y`: Training target data batch.
  - `val_batch_x`: Validation input data batch.
  - `val_batch_y`: Validation target data batch.
  - `df_val`: DataFrame for validation data.
  - `input_char_to_int`: Mapping from characters to integers for the input vocabulary.
  - `output_char_to_int`: Mapping from characters to integers for the output vocabulary.
  - `output_int_to_char`: Reverse mapping from integers to characters for the output vocabulary.
  - `beam_width`: Beam width for beam search.
  - `length_penalty`: Length penalty for beam search.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  - `max_length`: Maximum length of sequences.
  - `wandb_log`: Whether to log to Weights & Biases (1 for yes, 0 for no).

- **Returns**:
  - `model`: The trained Seq2Seq model.
  - `beam_val`: Validation accuracy using beam search.

### beam_search()
- **Arguments**:
  - `model`: The Seq2Seq model for inference.
  - `input_seq`: Input sequence for translation.
  - `max_length`: Maximum length of the input sequence.
  - `input_char_index`: Mapping from characters to integers for the input vocabulary.
  - `output_char_index`: Mapping from characters to integers for the output vocabulary.
  - `reverse_target_char_index`: Reverse mapping from integers to characters for the output vocabulary.
  - `beam_width`: Beam width for beam search.
  - `length_penalty`: Length penalty for beam search.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Returns**:
  - `str`: The generated output sequence.

---

## Installation

To run the training script, ensure you have Python 3 installed along with the following dependencies:

- torch
- numpy
- pandas
- tqdm
- wandb
- argparse

You can install these dependencies using pip:

```bash
pip install torch numpy pandas tqdm wandb argparse
```

## Usage

To train the Seq2Seq model with different RNN cell types, use the `train.py` script with the following command-line arguments:

| Argument             | Shorthand | Type   | Default Value | Choices                   | Description                                                  |
|----------------------|-----------|--------|---------------|---------------------------|--------------------------------------------------------------|
| --data_path          | -dp       | str    | Dakshina_sampled           |                           | Path to the data folder.                                    |
| --lang               | -l        | str    | 'hin'         |                           | Language for which training is to be done.                   |
| --embedding_size     | -es       | int    | 256           |                           | Embedding size.                                              |
| --hidden_size        | -hs       | int    | 512           |                           | Hidden size.                                                 |
| --num_layers         | -nl       | int    | 2             |                           | Number of layers.                                           |
| --cell_type          | -ct       | str    | 'LSTM'        | RNN, LSTM, GRU            | Cell type (RNN, LSTM, GRU).                                  |
| --dropout            | -dr       | float  | 0.3           |                           | Dropout rate.                                                |
| --learning_rate      | -lr       | float  | 0.01          |                           | Learning rate.                                               |
| --batch_size         | -bs       | int    | 32            |                           | Batch size.                                                  |
| --num_epochs         | -ep       | int    | 10            |                           | Number of epochs.                                           |
| --optimizer          | -op       | str    | 'adagrad'     | adam, sgd, rmsprop,       | Optimizer (adam, sgd, rmsprop, nadam, adagrad).             |
|                      |           |        |               | nadam, adagrad             |                                                              |
| --beam_search_width  | -bw       | int    | 1             |                           | Beam search width.                                           |
| --length_penalty     | -lp       | float  | 0.6           |                           | Length penalty.                                              |
| --teacher_forcing    | -tf       | float  | 0.7           |                           | Teacher forcing ratio.                                       |
| --bidirectional      | -bi       | bool   | True          | True, False               | Use bidirectional encoder.                                   |
| --store_csv          | -store    | bool   | False         | True, False               | Store Prediction CSV in current directory.                    |
| --test_accuracy      | -test     | int    | 0             | 0, 1                      | Flag to enable test accuracy.                                |
| --wandb_log          | -wl       | int    | 0             | 0, 1                      | Whether to log to WandB (1 for yes, 0 for no).               |
| --wandb_project      | -wp       | str    | DL_Assignment_3 |                           | Project name used to track experiments in Weights & Biases.  |
| --wandb_entity       | -we       | str    | cs24m019-iitm |                           | Wandb Entity used to track experiments in Weights & Biases.  |


Example command to run the training script:

```bash
python train_vanilla.py -dp /path/up/to/Dakshina_sampled -l hin -es 256 -hs 512 -nl 2 -ct LSTM -ep 10 -bs 32 -op adagrad -bi True -bw 1 -test 1
```

```
**Must Follow this guidelines**
1) Make Sure to run on GPU to enable fast training .
2) Make Sure to give path up to sample language folder i.e. -dp /path/up/to/language folder/ and then choose language, i.e., -l hin,guj,etc (Dont give train, val, test path since script will handle it).
3) Select flag -test 1 to print test accuracy at word level in last.
```

## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy Char**: The character-level accuracy of predictions on the training data.
- **Train Average Loss**: The average loss calculated during training.
- **Validation Accuracy Char**: The character-level accuracy of predictions on the validation data.
- **Validation Average Loss**: The average loss calculated during validation.
- **Beam Val Word Accuracy**: The word-level accuracy of predictions on the validation data using beam search.
- **Test Accuracy Word Level**: The word-level accuracy of predictions on the test data using naive method i.e. argmax only.
- **Correct Prediction**: The number of correct predictions out of the total data samples.

These metrics provide insights into the performance of the Seq2Seq model during training and validation. Character-level accuracy measures how accurately the model predicts individual characters, while word-level accuracy assesses the correctness of entire output sequences.


---




# BASE Seq2Seq MODEL WITH ATTENTION

This repository contains a Python implementation of a sequence-to-sequence (Seq2Seq) model with attention mechanism for sequence prediction tasks. The Seq2Seq model is implemented using PyTorch and includes an attention mechanism to focus on relevant parts of the input sequence during decoding.


## Usage

To use the Seq2Seq model with attention, follow the steps below:

1. Import the necessary classes from the provided code.

2. Initialize an instance of the `Encoder` class with the required parameters:
   - `input_size`: Size of the input vocabulary.
   - `embedding_size`: Size of the embedding layer.
   - `hidden_size`: Size of the hidden state in the RNN.
   - `num_layers`: Number of layers in the RNN.
   - `dropout`: Dropout rate for regularization.
   - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

3. Initialize an instance of the `Attention` class with the required parameter:
   - `hidden_size`: Size of the hidden state.

4. Initialize an instance of the `Decoder` class with the required parameters:
   - `input_size`: Size of the input vocabulary.
   - `embedding_size`: Size of the embedding layer.
   - `hidden_size`: Size of the hidden state in the RNN.
   - `output_size`: Size of the output vocabulary.
   - `num_layers`: Number of layers in the RNN.
   - `dropout`: Dropout rate for regularization.
   - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

5. Initialize an instance of the `Seq2Seq` class with the required parameters:
   - `encoder`: Instance of the Encoder class.
   - `decoder`: Instance of the Decoder class.
   - `target_vocab_size`: Size of the target vocabulary.
   - `teacher_force_ratio`: Ratio of teacher forcing during training.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

6. Train the Seq2Seq model using the `train()` function with the required arguments:
   - `model`: The Seq2Seq model to be trained.
   - `num_epochs`: Number of training epochs.
   - `criterion`: Loss criterion for training.
   - `optimizer`: Optimizer for training.
   - `train_batch_x`: Training input data batch.
   - `train_batch_y`: Training target data batch.
   - `val_batch_x`: Validation input data batch.
   - `val_batch_y`: Validation target data batch.
   - `df_val`: DataFrame for validation data.
   - `input_char_to_int`: Mapping from characters to integers for the input vocabulary.
   - `output_char_to_int`: Mapping from characters to integers for the output vocabulary.
   - `output_int_to_char`: Reverse mapping from integers to characters for the output vocabulary.
   - `beam_width`: Beam width for beam search.
   - `length_penalty`: Length penalty for beam search.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
   - `max_length`: Maximum length of sequences.
   - `wandb_log`: Whether to log to Weights & Biases (1 for yes, 0 for no).

## Classes

### Encoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the encoder.
  - `forward()`: Performs forward pass through the encoder.

### Attention
- **Data Members**:
  - `hidden_size`: Size of the hidden state.
  
- **Methods**:
  - `__init__()`: Initializes the Attention mechanism.
  - `dot_score()`: Calculates the dot product attention scores between the decoder hidden state and encoder outputs.
  - `forward()`: Performs forward pass through the Attention mechanism.

### Decoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `output_size`: Size of the output vocabulary.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the decoder.
  - `forward()`: Performs forward pass through the decoder.

### Seq2Seq
- **Data Members**:
  - `encoder`: Instance of the Encoder class.
  - `decoder`: Instance of the Decoder class.
  - `target_vocab_size`: Size of the target vocabulary.
  - `teacher_force_ratio`: Ratio of teacher forcing during training.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Methods**:
  - `__init__()`: Initializes the Seq2Seq model.
  - `forward()`: Performs forward pass through the model.


## Command-line Arguments

| Argument             | Shorthand | Type   | Default Value | Choices                   | Description                                                  |
|----------------------|-----------|--------|---------------|---------------------------|--------------------------------------------------------------|
| --data_path          | -dp       | str    | Dakshina_sampled           |                           | Path to the data folder.                                    |
| --lang               | -l        | str    | 'hin'         |                           | Language for which training is to be done.                   |
| --embedding_size     | -es       | int    | 256           |                           | Embedding size.                                              |
| --hidden_size        | -hs       | int    | 512           |                           | Hidden size.                                                 |
| --num_layers         | -nl       | int    | 2             |                           | Number of layers.                                           |
| --cell_type          | -ct       | str    | 'LSTM'        | RNN, LSTM, GRU            | Cell type (RNN, LSTM, GRU).                                  |
| --dropout            | -dr       | float  | 0.3           |                           | Dropout rate.                                                |
| --learning_rate      | -lr       | float  | 0.01          |                           | Learning rate.                                               |
| --batch_size         | -bs       | int    | 32            |                           | Batch size.                                                  |
| --num_epochs         | -ep       | int    | 10            |                           | Number of epochs.                                           |
| --optimizer          | -op       | str    | 'adagrad'     | adam, sgd, rmsprop,       | Optimizer (adam, sgd, rmsprop, nadam, adagrad).             |
|                      |           |        |               | nadam, adagrad             |                                                              |
| --beam_search_width  | -bw       | int    | 1             |                           | Beam search width.                                           |
| --length_penalty     | -lp       | float  | 0.6           |                           | Length penalty.                                              |
| --teacher_forcing    | -tf       | float  | 0.7           |                           | Teacher forcing ratio.                                       |
| --bidirectional      | -bi       | bool   | True          | True, False               | Use bidirectional encoder.                                   |
| --store_csv          | -store    | bool   | False         | True, False               | Store Prediction CSV in current directory.                    |
| --heatmap_plot       | -hmap     | int    | 0             | 0, 1                      | Flag to enable heatmap plot.                                 |
| --test_accuracy      | -test     | int    | 0             | 0, 1                      | Flag to enable test accuracy.                                |
| --wandb_log          | -wl       | int    | 0             | 0, 1                      | Whether to log to WandB (1 for yes, 0 for no).               |
| --wandb_project      | -wp       | str    | DL_Assignment_3 |                           | Project name used to track experiments in Weights & Biases.  |
| --wandb_entity       | -we       | str    | cs24m019-iitm |                           | Wandb Entity used to track experiments in Weights & Biases.  |


## Example Usage

```bash
python train_attention.py -dp /path/up/to/Dakshina_sampled -l hin -es 256 -hs 512 -nl 2 -ct LSTM -ep 10 -bs 32 -op adagrad -bi True -bw 1 -test 1 
```

```
**Must Follow this guidelines**
1) Make Sure to run on GPU to enable fast training.
2) Make Sure to give path up to sample language folder i.e. -dp /path/up/to/language folder/ and then choose language, i.e., -l hin,guj,etc (Dont give train, val, test path since script will handle it).
3) Select flag -test 1 to print test accuracy at word level in last.
4) Heat Map will execute but not plot the heatmap on kernel if backend not supported so on kaggle it will not work but could work on VS.
```
## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy Char**: The character-level accuracy of predictions on the training data.
- **Train Average Loss**: The average loss calculated during training.
- **Validation Accuracy Char**: The character-level accuracy of predictions on the validation data.
- **Validation Average Loss**: The average loss calculated during validation.
- **Beam Val Word Accuracy**: The word-level accuracy of predictions on the validation data using beam search.
- **Test Accuracy Word Level**: The word-level accuracy of predictions on the test data using naive method i.e. argmax only.
- **Correct Prediction**: The number of correct predictions out of the total data samples.

These metrics provide insights into the performance of the Seq2Seq model during training and validation. Character-level accuracy measures how accurately the model predicts individual characters, while word-level accuracy assesses the correctness of entire output sequences.

