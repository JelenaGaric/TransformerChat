# TransformerChat
Chat bot using a transformer model. Done in tensorflow.


Calculating attention:


![alt text](https://4.bp.blogspot.com/-JGovZjJGRdU/XfgOi02BVUI/AAAAAAAAB04/hYZsLatBIkE30aYDftH7avQ6dL4KyJ3KgCLcBGAsYHQ/s1600/formula.png)

Where Q, K, V are vectors representing queries, keys and values. Query is a vector representing a word which we want to get attention values for, 
keys is a context (sentence for example), and values is representing

For each word, we do a dot product of query and transposed keys, and then normalise the scalar. Those normalized weights are then being multiplied by value vectors 
(word vectors of the context) to get more contextualized word embeddings (weighted average of original vectors on the input).




