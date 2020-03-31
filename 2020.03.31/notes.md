# MLNN

## Inductive Bias

### We have seen

* Basics supervised ML
* LM
* perceptronse

### Deep learning - Representations

* Deep learning models can learn the features itself
* Alleviating some of the need for domain specific knowledge
* State of the art

### Inductive biases

* Convolution
* Sequntial calculation

* If we have multiple possible sols, we want the simplest one. 
* Complicated functions can over fit
* So we want a simpler model - Occams razor
* Try the simpler before the more complex ones

### Inductive Bias - Local invariance

* Need: Roust to local changes, but not for global changes




## Convolutional NN

* Picks up some changes
* Needs fewer parameters
* Learns to specialize 
* Needs multiple kernels, usually
* Identety, edge detection, high contrast, ...
* Good for classification
* Text, speech
* Image is deep in layers, but for speech is feew layers

## Sequentiality
* for text
* generating sequences
* current output depends on prew output

## Rcurent NN

* Update a mem cell with a funtion, using prew val and ...
* Incorperating new information to the prew value
* understanding sentences
* Generating text
* Encoder - Decoder
* machine translation
* simple models don't work in the real world
* to avvoid:
  * this we make long-short term mem networks
  * Gated Recurrent Units

* Under training
  * Use a variant of back propagation (- trough time)
  * gradient clipping, regularization

## Recursive NN

* Three and graph like structure
* Language is roughly hierarchical
* Social Networks

## Self-Attention Networks

* Directly incorporated interaction
* Can be parallelized
* Can be used for deeper layers
* A new model
* Transformer - Encoder - Decoder
* Question answering

## --

* We can have inductive biased in other models then neural models