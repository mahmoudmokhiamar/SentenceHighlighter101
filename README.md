# Sentence Highlighter with BERT Embeddings

This Python script utilizes BERT (Bidirectional Encoder Representations from Transformers) to highlight sentences based on their similarity using cosine similarity scores. It transforms text files into sentences, computes BERT embeddings, calculates similarities, and generates an HTML file with highlighted sentences.

## Features

- **File Reading**: Reads text files (`file1.txt`, `file2.txt`) and extracts sentences using NLTK's `sent_tokenize`.
- **Sentence Transformation**: Converts sentences into BERT embeddings using the `bert-base-uncased` model.
- **Similarity Calculation**: Computes cosine similarity between sentence embeddings to determine sentence similarities.
- **HTML Generation**: Generates an HTML file (`highlighted_sentences.html`) where sentences are color-coded based on their similarity.

## Setup Instructions

### Install Dependencies

1. Install Python dependencies using `pip install -r requirements.txt`.
2. Ensure `nltk`, `transformers`, `torch`, `numpy`, and `scikit-learn` are installed.

### Run the Script

1. Modify the file paths in `read_files` to point to your specific text files.
2. Execute the script using Python: `python sentence_highlighter.py`.
3. This will generate an HTML file (`highlighted_sentences.html`) in the current directory.

## Usage

- **Customization**: Modify `model_name` in `sentenceHighlighter` initialization to use different BERT models or adjust color saturation and brightness in `calculate_similarity`.
- **Visualization**: Open `highlighted_sentences.html` in a web browser to view sentences color-coded by their similarity.

## Example

```python
sentenceHighlighter = sentenceHighlighter()
sentenceHighlighter.read_files(["file1.txt", "file2.txt"])
sentenceHighlighter.transform_sentences()
sentenceHighlighter.calculate_similarity()
sentenceHighlighter.generate_html()
