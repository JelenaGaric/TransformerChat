# TransformerChat
Chat bot using a transformer model. Done in tensorflow.


Calculating attention:


![alt text](https://glassboxmedicine.files.wordpress.com/2019/08/attention-equation.png)

Where Q, K, V are vectors representing queries, keys and values. Query is a vector representing a word which we want to get attention values for, 
keys is a context (sentence for example), and values is representing

For each word, we do a dot product of query and transposed keys, and then normalise the scalar. Those normalized weights are then being multiplied by value vectors 
(word vectors of the context) to get more contextualized word embeddings (weighted average of original vectors on the input).




