# Medical X-ray Image Captioning with Attention-based Encoder-Decoder

This project implements an attention-based encoder-decoder model to generate descriptive captions for medical X-ray images. It uses a pre-trained CheXNet model (DenseNet121) to encode image features and a GRU-based decoder with global attention to produce captions. The model processes two X-ray images simultaneously, making it suitable for comparative analysis in medical diagnostics.

## Project Overview

- **Goal**: Automate the generation of captions for medical X-ray images to assist radiologists in reporting and diagnosis.
- **Approach**: An encoder-decoder framework where the encoder extracts features from two X-ray images using CheXNet, and the decoder generates captions using a GRU with global attention.
- **Highlights**:
  - Dual-image input for enhanced context.
  - Attention mechanism to focus on key image regions.
  - Pre-trained CheXNet for robust medical image feature extraction.

## Model Architecture

### Image Encoder
- **Input**: Two X-ray images, each resized to 224x224 pixels (RGB).
- **Backbone**: Pre-trained CheXNet (DenseNet121) extracts features, outputting `(None, 7, 7, 1024)` per image.
- **Pooling**: Average pooling reduces features to `(None, 3, 3, 1024)`, reshaped to `(None, 9, 1024)`.
- **Dense Layer**: Reduces dimensionality to `(None, 9, 512)` per image.
- **Combination**: Features are concatenated to `(None, 18, 512)`, followed by batch normalization and dropout (rate=0.2).

### Decoder
- **Input**: Encoded image features and a token sequence (caption).
- **Embedding**: Tokens are embedded into 300-dimensional vectors.
- **Attention**: Global attention computes a context vector based on image features and the decoder state.
- **GRU**: A 512-unit GRU processes the context and embedded token to update the hidden state.
- **Output**: A dense layer predicts the next token’s probability distribution.

### One-Step Decoder
- Generates one token per step using attention and GRU.
- Updates the hidden state and predicts the next token.

### Overall Model
- **Training**: Uses teacher forcing with true captions.
- **Inference**: Generates captions via greedy or beam search, starting with `<cls>` and stopping at `<end>` or a max length of 29 tokens.

## Dataset
- Trained on the Indiana University Chest X-ray dataset with 7,477 X-ray images (frontal and lateral projections) and  4,169 patient captions.


## Future work
This file contains the architecture of a machine learning model, which can be fine-tuned and enhanced. To improve the model, I suggest the following:

- **Utilize BERT for embedding layers**: This leverages BERT's pre-trained language understanding capabilities, providing a robust foundation for natural language processing tasks.
- **Incorporate additional diverse datasets for training**: More varied data can improve the model's generalization and performance by helping it learn a broader range of patterns and reducing overfitting.
- **Optimize training efficiency with C programming language**: Implementing performance-critical components of the model or training process in C can enhance speed, as C offers faster execution for computationally intensive tasks.
- **Leverage the Fastai library**: This can streamline model development, training, and experimentation, potentially boosting performance through Fastai’s high-level abstractions and deep learning best practices.
