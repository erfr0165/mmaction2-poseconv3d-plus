## File Explanation

| File Name                        | Description                                                                                              |
|----------------------------------|----------------------------------------------------------------------------------------------------------|
| **`download_embeddings.bash`**   | Executes download and extraction of fastText (300-d) Word Embeddings.                                    |
| **`create_vocabulary.py`**       | Responsible for collecting a few hundred to many thousand words and their corresponding word embeddings. |
| **`encoder.py`**                 | Dimension reduction of word embeddings with neural encoder.                                              |
| **`pca.py`**                     | Dimension reduction of word embeddings with PCA.                                                         |
| **`write_embeddings_to_pkl.py`** | Collects embeddings from relevant words for action recognition.                                          |
| **`visualization.py`**           | Performs visualization of word embeddings based on cosine similarity.                                    |
| **`loss_computation.py`**        | Calculates the loss for the reduced word embeddings based on cosine similarity.                          |