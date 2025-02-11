# Word2Vec Training on High-Quality English Sentences

## Overview
This repository contains an end-to-end pipeline for processing and training a Word2Vec model on high-quality English sentences. The pipeline includes text preprocessing techniques such as tokenization, stopword removal, regex-based cleaning, and lemmatization, followed by training a continuous bag-of-words (CBOW) Word2Vec model to generate word embeddings.

## Dataset
We use the **agentlans/high-quality-english-sentences** dataset, which is loaded using the `datasets` library. A subset of 10,000 sentences is extracted and processed for training the Word2Vec model.

## Techniques Used
### 1. Text Tokenization
- The **UnicodeScriptTokenizer** from `tensorflow_text` is used to tokenize sentences into words efficiently.
- Tokenized words are extracted and decoded for further processing.

### 2. Text Preprocessing
- **Stopword Removal:** NLTK's predefined English stopword list is used to filter out common words that do not contribute significantly to the meaning.
- **Regex-based Cleaning:** Regular expressions (`re.sub()`) are applied to remove non-alphabetic characters, ensuring that only meaningful words remain.
- **Lemmatization:** NLTK’s `WordNetLemmatizer` is used to reduce words to their base form, improving the consistency of text data.

### 3. Word2Vec Model Training
- The **Gensim Word2Vec** model is trained on the preprocessed data.
- We use the **Continuous Bag-of-Words (CBOW)** architecture (`sg=0`), which is effective for learning meaningful word relationships.
- Hyperparameters:
  - `vector_size=100`: Each word is represented by a 100-dimensional vector.
  - `window=3`: Words within a context window of 3 words are considered.
  - `min_count=2`: Words appearing less than twice are ignored.
  - `epochs=20`: The model is trained for 20 iterations.
  - `hs=1`: Hierarchical softmax is used for optimization.

### 4. Word Embedding Evaluation
- The trained model is saved as `word2vec.model` for future use.
- The word vector for the term **"social"** is extracted and printed to showcase how the model represents words in high-dimensional space.
- The most similar words to **"social"** are retrieved using the `most_similar()` function, demonstrating the model’s ability to learn semantic relationships.

## Dependencies
To execute this project, install the following dependencies:
```bash
pip install datasets tensorflow tensorflow-text nltk gensim
```
Additionally, download the required NLTK resources before running the script:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Execution
Run the script to process the text, train the model, and evaluate embeddings:
```bash
python train_word2vec.py
```

## Results
After execution, the model generates meaningful word vectors that can be used for various NLP tasks such as text classification, sentiment analysis, and similarity detection. The trained model can be loaded and used for further analysis or downstream applications.

## Future Enhancements
- Implement **Skip-gram** (`sg=1`) for better performance on infrequent words.
- Use **pre-trained embeddings** such as GloVe or FastText for comparison.
- Fine-tune hyperparameters for improved results.
- Extend preprocessing with stemming and POS tagging for better linguistic representation.

## License
This project is open-source and available under the MIT License.

