import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from chatbot.qa_data import questions, answers  # Import question and answer data

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

processed_questions = [preprocess_text(question) for question in questions]
processed_answers = [preprocess_text(answer) for answer in answers]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_questions + processed_answers)
vocab_size = len(tokenizer.word_index) + 1

# Create input and output sequences
input_sequences = tokenizer.texts_to_sequences(processed_questions)
input_sequences = pad_sequences(input_sequences, padding='post')

output_sequences = tokenizer.texts_to_sequences(processed_answers)
output_sequences = pad_sequences(output_sequences, padding='post')

# Add special tokens to tokenizer
tokenizer.word_index['<start>'] = vocab_size
tokenizer.index_word[vocab_size] = '<start>'
tokenizer.word_index['<end>'] = vocab_size + 1
tokenizer.index_word[vocab_size + 1] = '<end>'
tokenizer.word_index['<unknown>'] = vocab_size + 2
tokenizer.index_word[vocab_size + 2] = '<unknown>'
vocab_size += 3  # Increment vocab_size to account for the new token

# Define Seq2Seq model with attention
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# Define hyperparameters
embedding_dim = 256
units = 512
batch_size = 1

# Create the encoder and decoder
encoder = Encoder(vocab_size, embedding_dim, units)
decoder = Decoder(vocab_size, embedding_dim, units)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Training step
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

# Train the model
EPOCHS = 100

for epoch in range(EPOCHS):
    total_loss = 0
    enc_hidden = tf.zeros((batch_size, units))
    for batch in range(len(input_sequences)):
        inp = input_sequences[batch:batch+1]
        targ = output_sequences[batch:batch+1]
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(input_sequences):.4f}')

# Chat with the chatbot
def evaluate(sentence, max_response_length=50):
    sentence = preprocess_text(sentence)
    inputs = [tokenizer.word_index.get(i, tokenizer.word_index['<unknown>']) for i in sentence.split()]
    inputs = pad_sequences([inputs], maxlen=input_sequences.shape[-1], padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = []

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    for t in range(max_response_length):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        if (
            tokenizer.index_word[predicted_id] != '<start>' and
            tokenizer.index_word[predicted_id] != '<unknown>' and
            tokenizer.index_word[predicted_id] not in result
        ):
            result.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)