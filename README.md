# End2EndSpeechRecModel
An Attention-based Speech-to-Text Deep Encoder-Decoder Neural Network using pyramidal Bi-LSTMs

This model follows the LAS(Listen, Attend, Spell) framework.
The baseline model for this assignment is described in the Listen, Attend and Spell paper. The idea is to learn all components of a speech recognizer jointly. The paper describes an encoder-decoder approach, called Listener and Speller respectively.
The Listener consists of a Pyramidal Bi-LSTM Network structure that takes in the given utterances and compresses it to produce high-level representations for the Speller network.
The Speller takes in the high-level feature output from the Listener network and uses it to compute a probability distribution over sequences of characters using the attention mechanism.
Attention intuitively can be understood as trying to learn a mapping from a word vector to some areas of the utterance map. The Listener produces a high-level representation of the given utterance and the Speller uses parts of the representation (produced from the Listener) to predict the next word in the sequence.

HOW TO RUN:
The model was developed in the ipynb on AWS's ec2 server and Google Colab. The file can be run by executing the cells sequentially on a jupyter notebook or in colab.

Model description:
An end-to-end speech-to-text model was trained comprising of the following elements:

1. Data Loading
- The Speech2TextDataset class inherits from torch.util.data's DataSet class and is as such used to load the training,
  validation and test data.
- __len__ return the number of data samples (total number of utterances)
- __getitem__ accesses the utterance index 'i'. A labels tensor is returned corresponding to each frame of the input utterance.
- A custom collate_fn 'collate_train' is used to pad the utterances and labels before creating packed sequences for processing by the LSTMs. The collate function removes <sos> from the text labels.
- Length of the utterances and the labels are also returned as part of the collate_fn to facilitate the creation of packed sequences.


2. Network Architecture
- The Seq2Seq model comprises of an Encoder, Decoder and Attention object. Here hiddendim = keysize = valuesize = 256

Attention:
- The attention class performs bmm between the input key and query to generate energy. Dimensions of query were manipulated to perform the bmm.
- A binary mask was implemented to remove the padded values from the encoder’s output. Since I am normalizing the output using softmax, the padded values were masked with large negative values which zero out after softmax. This softmax operation gives attention.
- The masked attention is again batch matrix multiplied with value to give context. Both context and masked attention weights are returned

Encoder:
- The encoder uses a simple lstm and 3 pblstms. The hidden dimension for both is 256. 
- The input dimension for the simple lstm is 40 and for the pblstm = 4 *hidden_dim
- Both lstm use locked dropout with p = 0.3. I also experimented with weight-connect but it did not help.

Decoder:
- The decoder has an embedding layer with input_dim = vocab size, and output = hidden_dim
- This is followed by 2 lstm cells. The first one has dimensions (hidden_dim+value_size, 2*hidden_dim). The second one has dimensions (2*hidden_dim, key_size)
- The output of these lstms is used to generate attention and context at each time step. This output and context are passed through a combination of linear+elu each before making the prediction using the character_prob linear layer (key_size + value_size, vocab size). 
- The character_prob layer's weights are tied to the embedding layer's weights

- Greedy search is used to decode. Random search is implemented but it is not used as I couldn't get it to work. :(


3. Hyperparameters and implementation details
- epochs = 45
  I ran the model for 45 epochs and checked the validation Lev distance every epoch. 
  The validation accuracy was measured using Levenshtein distance between the label sequence and the greedily predicted sequence.
  Best results were obtained from epoch 34.

- training batch_size = 64
  Batch sizes of 32 and 64 were experimented with. Increasing the batch size sped up the computation due to vectorized implementation.
  With a batch size of 124 and above however, the GPU ran out of memory. The model was thus trained with a batch size of 64.

- Optimizer = Adam
  Adam with default learning rate of 1e-3 was used to optimize the weight updates.
  The learning rate was annealed using a ReduceLROnPlateau scheduler with a factor = 0.3, patience=2 and threshold=1e-2. Factors of 0.5 and 0.1 were experimented with. The learning rate was annealed based on the validation Lev distance

- Criterion = CTCLoss
  The CTCLoss criterion was used to make use of the CTC algorithm to decode the output sequence with reduce = 'none'
  The predictions and labels were binary masked before computing the loss. The padded values were zeroed out. The loss was summed over all characters and sequences and divided by the sum of 1s in the mask.
  
- Levenshtein Distance
  The Levenshtein distance was computed between the decoded sequence and the input labels to compute the accuracy. 
  The Lev distance stopped reducing after 40 epochs and stabilized at around 19.

- Teacher forcing was used and scheduled as follows:
    if epoch < 10:
        tf = 0.9
    elif epoch >= 10 and epoch < 20:
        tf = 0.8
    elif epoch >= 20 and epoch < 25:
        tf = 0.75
    else:
        tf = 0.7
