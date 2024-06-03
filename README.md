# ML model partitioning
Simulation of model partitioning and parallelism scenarios based on IMDB dataset.

## 1. Data Parallelism
- data_parallelism_train.py
- data_parallelism_test.py

Data parallelism can be achieved by making a copy of a model, passing the split input to parallelly available models, and aggregating the result. 

## 2. Model Parallelism
- model_parallelism_train.py
- model_parallelism_test.py

Model parallelism can be achieved by training multiple models on a split training dataset. When input data for inference is available, it needs to be sent to all the parallel available models and aggregated. 

## 3. Model Partitioning for Pipeline Parallelism 
- model_partitioning.py

Model partitioning (for Pipeline parallelism) can be achieved by splitting the trained model into multiple partitions based on layers in sequence; for instance, if the model has seven layers, the first two layers can be model partition 1, the next three layers can be model partition 2, and the last two layers can be model partition 3.
When an input for inference is present, it is first executed against model partition 1. The result is then input to the second model partition, which is then input to model partition 3. Finally, the final result is obtained from model partition 3. 

## 4. LSTM Classifier
- lstm-for-classification.py

This is done extra to try out a multi-layer neural network for model partition. 
