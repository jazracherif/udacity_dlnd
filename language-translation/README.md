# Machine Translation using LSTMs

In this project, our goal is to translate sentences from English to French. I use a simple Encode-Decoder Architecture and use Tensorflow contrib modules for the implementation

To build the **Encoder**, I wrap a single LSTM cell in a multiRNNCell structure and use dropout:
```
tf.contrib.layers.embed_sequence
tf.contrib.rnn.LSTMCell
tf.contrib.rnn.DropoutWrapper
tf.contrib.rnn.MultiRNNCell
tf.nn.dynamic_rnn
```

To build the **Decoder**, I build 2 modules, one for training and the other for inferring. I rely on the following functions:
```
tf.contrib.seq2seq.GreedyEmbeddingHelper
tf.contrib.seq2seq.BasicDecoder
tf.contrib.seq2seq.dynamic_decode
```
