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




