import argparse
import numpy as np
import os
import torch

from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description='Parameter for visualization')
    parser.add_argument(
        '--vocabulary-path',
        type=str,
        default='./vocabulary.txt',
        help='path to embeddings that are used in reduction')
    parser.add_argument(
        '--output-file-path',
        type=str,
        default='./embeddings.txt',
        help='path where to save reduced embeddings')
    parser.add_argument(
        '--dimension',
        type=int,
        default=20,
        help='dimension the word embeddings should get reduced to')

    args = parser.parse_args()
    return args


# code from 'https://github.com/vyraun/Half-Size/blob/master/algo.py'
def reduce_dim_pca(emb_dict: dict, new_dim: int, output_path: str):
    """Performs PCA and writes reduced word embeddings to output file.

        Args:
            emb_dict (dict): Dictionary with words and original high-dimensional word embeddings.
            new_dim (int): New Dimension after PCA.
            output_path (str): Path to output file.
    """
    X_train = []
    X_train_names = []
    for x in emb_dict:
        X_train.append(emb_dict[x])
        X_train_names.append(x)

    X_train = np.asarray(X_train)

    # PCA to get Top Components
    pca =  PCA(n_components = 300)
    X_train = X_train - np.mean(X_train)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
        for u in U1[0:7]:        
            x = x - np.dot(u.transpose(),x) * u 
        z.append(x)

    z = np.asarray(z)

    # PCA Dim Reduction
    pca =  PCA(n_components = new_dim)
    X_train = z - np.mean(z)
    X_new_final = pca.fit_transform(X_train)


    # PCA to do Post-Processing Again
    pca =  PCA(n_components = new_dim)
    X_new = X_new_final - np.mean(X_new_final)
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    X_new_final = X_new_final - np.mean(X_new_final)

    final_pca_embeddings = {}
    embedding_file = open(output_path, 'w')

    # write embeddings to output file
    for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write("%s\t" % x)
        # remove second post processing from original code
        # reason: first ( Dimensions are basically zeros and don't contain information)
        # for u in Ufit[0:7]:
        #     final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 

        for t in final_pca_embeddings[x]:
            embedding_file.write("%f\t" % t)
        
        embedding_file.write("\n")


def main():
    args = parse_args()

    word_to_embedding = {}
    # read embeddings that are stored in .txt file
    with open(args.vocabulary_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            w = True
            values = line.split()
            word = values[0]
            i = 1
            while w:
                # check if entry is number, otherwise it still belongs to the embedded word(s)
                try:
                    vector = torch.tensor([float(val) for val in values[i:]], dtype=torch.float)
                    w = False
                except ValueError:
                    word += ' '
                    word += values[i]
                    i +=1
            word_to_embedding[word] = np.asarray(vector)
    
    # check if output directory exists
    directory = os.path.dirname(args.output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # perform pca
    reduce_dim_pca(word_to_embedding, args.dimension, args.output_file_path)


if __name__ == '__main__':
    main()