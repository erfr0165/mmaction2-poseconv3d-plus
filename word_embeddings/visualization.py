import argparse
import fasttext
import fasttext.util
import matplotlib.pyplot as plt 
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn.functional as F


# example words to visualize the cosine similarity of their embeddings
SMALL = ('knee', 'ankle', 'shoulder', 'elbow', 'wrist', 'hammer', 'wrench', 'screwdriver', 'chisel', 'pliers')
FULL = ('eye', 'left-eye', 'right-eye', 'ear', 'nose', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'spine',    
	'hammer', 'chisel', 'wrench', 'screwdriver', 'pliers',
	'table', 'board',  
	'left', 'right')


def parse_args():
    parser = argparse.ArgumentParser(description='Parameter for visualization')
    parser.add_argument(
        '--embeddings',
        type=str,
        default='fasttext',
        choices=['fasttext', 'custom'],
        help='which embeddings to use for visualization')
    parser.add_argument(
        '--output-file-path',
        type=str,
        default='./cosine_similarity_matrix.png',
        help='path where to save output matrix')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./fasttext_embeddings/cc.en.300.bin',
        help='path to downloaded fasttext model (needed when embeddings=fasttext)')
    parser.add_argument(
        '--custom-embeddings-path',
        type=str,
        default='./embeddings.txt',
        help='path to custom embeddings (needed when embeddings=custom)')
    parser.add_argument(
        '--words',
        type=str,
        default='full',
        choices=['full', 'small', 'custom'],
        help='determines which word to use for matrix')
    parser.add_argument(
        '--dimension',
        type=int,
        default=300,
        help='dimension of word embeddings to visualize (must be <=300)')
    parser.add_argument(
        '--custom-words-path',
        type=str,
        default='./custom_words.txt',
        help='path to .txt with words of which the cosine similarity of their embeddings should get visualized')

    args = parser.parse_args()
    return args


def plot_cosine_similarity(embeddings: list, words: list, out_path: str, size: str='full'):
    """Plot and save cosine similarity matrix.

        Args:
            embeddings (list): List of embeddings for which the similarity should be calculated.
            words (list): List of corresponding words to the embeddings.
            output_path (str): Path to output file.
            size (str): Size of matrix.
    """
    # compute similarity
    dimension = embeddings[0].shape[0]
    emb_tensor = torch.stack(embeddings)
    emb_tensor = emb_tensor.view(-1, dimension)
    emb_sim = F.cosine_similarity(emb_tensor.unsqueeze(1), emb_tensor.unsqueeze(0), dim=-1)
    # round values to 2 decimal places
    emb_sim = torch.round(emb_sim * 100) / 100
    if size=='small':
        plt.figure(figsize=(14,12))
    else:
        plt.figure(figsize=(26,24))
    sns.heatmap(emb_sim, annot=True, xticklabels=words, yticklabels=words, vmin=0, vmax=1, 
                annot_kws={"fontsize": 20},cbar_kws={'shrink': 0.7, 'ticks': [0, 0.25, 0.5, 0.75, 1], 'label': '', 'fraction': 0.046, 'pad': 0.04})

    plt.xticks(fontsize=26, rotation=45, ha='center')
    plt.yticks(fontsize=26, rotation=0, va='center')
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=24)

    plt.tight_layout()
    plt.savefig(out_path, format='png')
    plt.show()


def main():
    args = parse_args()

    if args.words == 'full':
        size = args.words
        words = FULL
    elif args.words == 'small':
        size = args.words
        words = SMALL
    elif args.words == 'custom':
        size = 'full'
        words = []
        with open(args.custom_words_path, 'r') as file:
            for line in file:
                words.append(line.strip())


    if args.embeddings == 'fasttext':
        ft = fasttext.load_model(args.model_path)
        if args.dimension < 300:
            # reduce fasttext model
            fasttext.util.reduce_model(ft, args.dimension)
        if args.dimension > 300 or args.dimension < 1:
            raise ValueError('Dimension must be between 1 and 300')
        embeddings = list()
        dimensions = len(ft.get_word_vector('test'))

        # get embeddings for all words
        for word in words:
            word_ = word.replace('_', ' ').replace('-', ' ').replace(':', ' ').replace('(', '').replace(')', '').split()
            if len(word_) > 1:
                emb = np.zeros((dimensions,), dtype=np.float32)
                for e in word_:
                    emb += ft.get_word_vector(e)
                emb /= len(word_)
                emb = torch.tensor(emb)
            else:
                emb = torch.tensor(ft.get_word_vector(word))
            embeddings.append(emb)


    elif args.embeddings == 'custom':
        word_to_embedding = {}
        # read embeddings that are stored in .txt file
        with open(args.custom_embeddings_path, 'r', encoding='utf-8') as f:
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
                word_to_embedding[word] = vector

        # get embeddings for the relevant words
        embeddings = []
        for word in words:
            word_ = word.replace('-', ' ')
            try:
                emb = word_to_embedding[word_]
            except KeyError:
                raise KeyError(f"Word Embedding for '{word_}' not found.")
            emb = torch.tensor(emb)
            embeddings.append(emb.cpu())

    directory = os.path.dirname(args.output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_cosine_similarity(embeddings, words, args.output_file_path, size)


if __name__ == '__main__':
    main()