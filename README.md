# Twitter Sentiment Analysis

This project utilizes LSTM (Long Short-Term Memory) networks and GloVe (Global Vectors for Word Representation) Word Embedding Vectors for analyzing sentiment on Twitter data.

## Installation

Before running the sentiment analysis, ensure you have the necessary dataset and dependencies installed.

### Dataset

The GloVe word embedding vectors can be downloaded and unzipped using the following command:

```bash
curl -O http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip -d path/to/destination
```
### Dependencies
This project is compatible with CUDA 11.2 for GPU acceleration on Native Windows environments and requires TensorFlow version 2.10.0. Ensure you have the correct TensorFlow versions installed by running:
```bash
pip install tensorflow==2.10.0 tensorflow-gpu==2.10.0
```
