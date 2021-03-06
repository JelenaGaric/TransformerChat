# TransformerChat
Chat bot using a transformer model. Done in tensorflow.


Calculating attention:


![alt text](https://glassboxmedicine.files.wordpress.com/2019/08/attention-equation.png)

Where Q, K, V are vectors representing queries, keys and values. Query is a vector representing a word which we want to get attention values for, 
keys is a context (sentence for example), and values is representing

For each word, we do a dot product of query and transposed keys. After we take the dot product, we divide by the square root of d_k
to prevent the dot products from getting huge as d_k (the vector length) increases. Then we normalise the scalar (applying softmax to squish 
into a (0,1) range - we now have attention weights.
Those normalized weights are then being multiplied by value vectors 
(word vectors of the context) to get more contextualized word embeddings (weighted average of original vectors on the input).



Positional encoding:

![alt text](https://glassboxmedicine.files.wordpress.com/2019/08/positional-encoding.png)

Where pos is the position of a word in the sentence and i are indexes in the embedding dimension (i is range from 1 to length of positional encoding vector). 
Sine and cosine - model easily learns to attend by relative positions.
After adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, 
in the d-dimensional space.

Model:

![alt text](https://glassboxmedicine.files.wordpress.com/2019/08/figure1modified.png?w=616)

Consists of encoder and decoder. Each one is made up of more layers. Each EncoderLayer has two sub-layers: multi-headed self-attention and a feedforward layer.
Each DecoderLayer has three sub-layers: multi-headed self-attention, multi-headed encoder-decoder attention, and a feedforward layer.
At the end of the Decoder, a linear layer and a softmax are applied to the Decoder output to predict the next word.
The Encoder is run once. The Decoder is run multiple times, to produce a predicted word at each step
