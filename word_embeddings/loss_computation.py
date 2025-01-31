import argparse
import numpy as np
import torch
import fasttext
import fasttext.util
import torch.nn.functional as F


SMALL = ['knee', 'ankle', 'shoulder', 'elbow', 'wrist', 'hammer', 'wrench', 'screwdriver', 'chisel', 'pliers']
FULL = ['eye', 'left-eye', 'right-eye', 'ear', 'nose', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'spine',
	'hammer', 'chisel', 'wrench', 'screwdriver', 'pliers',
	'table', 'board',
	'left', 'right']
IKEA = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', \
             'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
IKEA_OBJECTS = ['table top', 'leg', 'shelf', 'side panel', 'front panel', 'bottom panel', 'rear panel']
ATTACH = ['pelvis', 'spine navel', 'spine chest', 'neck', 'left clavicle', 'left shoulder', 'left elbow', \
					'left wrist', 'left hand', 'left hand tip', 'left thumb', 'right clavicle', 'right shoulder', 'right elbow', \
					'right wrist', 'right hand', 'right hand tip', 'right thumb', 'left hip', 'left knee', 'left ankle', 'left foot', \
					'right hip', 'right knee', 'right ankle', 'right foot', 'head', 'nose', 'left eye', 'left ear', 'right eye', 'right ear']
ATTACH_OBJECTS = ['screwdriver', 'cabinet foot', 'wall spacer top', 'screw no head', 'wrench', 'board', 'screw with head', 'threaded tupe female', \
				  'threaded rod', 'manual', 'hammer', 'wall spacer mounting']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method',
        type=str,
        default='fasttext',
        choices=['fasttext', 'custom'],
        help='which embeddings to use for visualization')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./fasttext_embeddings/cc.en.300.bin',
        help='path to downloaded fasttext model (used as target)')
    parser.add_argument(
        '--reduced-embeddings-path',
        type=str,
        default='./embeddings.txt',
        help='path to reduced embeddings (needed when embeddings=custom)')
    parser.add_argument(
        '--words',
        type=str,
        default='all',
        choices=['all', 'small', 'full', 'ikea', 'attach', 'ikea_objects', 'attach_objects', 'custom'],
        help='words used in loss computation')
    parser.add_argument(
        '--custom-words-path',
        type=str,
        default='./custom_words.txt',
        help='path to .txt with words of which the loss of their reduced embeddings should get computed')
    parser.add_argument(
        '--atomic',
        action='store_true',
        default=False,
        help='terms like "left shoulder" will be handled like "left"+"shoulder"')
    parser.add_argument(
        '--reduced-dimension',
        type=int,
        default=16,
        help='dimension of reduced embeddings if method is "fasttext"')

    args = parser.parse_args()
    return args


def get_fasttext_embeddings(model, words, dimension):
    """Get (original) fasttext word embeddings.
        Args:
            words (list): List of words.
            ft_model (fasttextModel): Model to extract the embeddings from.
            dimension (int): Number of dimensions of the word embeddings.

        Returns:
            List of word embeddings.
    """
    embeddings = []
    for word in words:
        word_ = word.replace('_', ' ').replace('-', ' ').replace(':', ' ').replace('(', '').replace(')', '').split()
        if len(word_) > 1:
            emb = np.zeros((dimension,), dtype=np.float32)
            for e in word_:
                emb += model.get_word_vector(e)
            emb /= len(word_)
            emb = torch.tensor(emb)
        else:
            emb = torch.tensor(model.get_word_vector(word))
        embeddings.append(emb)
    return embeddings

def get_reduced_embeddings(path, words, atomic):
    """Get embeddings of reduced word embeddings.

        Args:
            words (list): List of words of which the embeddings should be collected.
            path (str): Path to .txt file which contains the word embeddings.
            atomic (bool): if True: combine embeddings for terms that consist of multiple words by
                summation of the atomic words.

        Returns:
            List of word embeddings.
    """
    word_to_embedding = {}
    # read embeddings that are stored in .txt file
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            w = True
            values = line.split()
            word = values[0]
            i = 1
            while w:
                # check if entry is number, otherwise it still belongs to the embedded word(s)
                try:
                    vector = np.array([float(val) for val in values[i:]], dtype=np.float32)
                    w = False
                except ValueError:
                    word += ' '
                    word += values[i]
                    i +=1
            word_to_embedding[word] = vector

    embeddings = []
    emb = next(iter(word_to_embedding.items()))
    size = len(emb[1])
    for word in words:
        word_ = word.replace('-', ' ')
        if not atomic:
            try:
                emb = word_to_embedding[word_]
            except KeyError:
                raise KeyError(f"Word Embedding for '{word_}' not found.")
            emb = torch.tensor(emb)
        else:
            emb = np.zeros(size)
            word_ = word_.split(' ')
            for w in word_:
                try:
                    emb += word_to_embedding[w]
                except KeyError:
                    raise KeyError(f"Word Embedding for '{w}' not found.")
                emb /= len(word_)
                emb = torch.tensor(emb)
        embeddings.append(emb)
    return embeddings


def compute_similarity(embeddings):
    """Computation of cosine similarity matrix for word embeddings.
        Args:
            embeddings (list): List of word embeddings.
        Returns:
            cosine similarity matrix
    """
    dimension = embeddings[0].shape[0]
    tensor_stack = torch.stack(embeddings).view(-1, dimension)
    return F.cosine_similarity(tensor_stack.unsqueeze(1), tensor_stack.unsqueeze(0), dim=-1)


def main():
    args = parse_args()

    if args.words in ['small', 'full', 'ikea', 'attach']:
        words = globals()[args.words.upper()]
    elif args.words in ['ikea_objects', 'attach_objects']:
        joints = args.words.replace('_objects', '')
        words = globals()[joints.upper()] + globals()[f'{args.words.upper()}']
    elif args.words == 'all':
        words = []
        for l in ['small', 'full', 'ikea', 'ikea_objects', 'attach', 'attach_objects']:
            words += globals()[l.upper()]
        words = list(set(words))
    elif args.words == 'custom':
        words = []
        with open(args.custom_words_path, 'r') as file:
            for line in file:
                words.append(line.strip())

    # get target word embeddings
    ft = fasttext.load_model(args.model_path)
    print('fasttext model loaded succesfully')
    targets = get_fasttext_embeddings(ft, words, 300)
    
    # get reduced word embeddings
    if args.method == 'fasttext':
        fasttext.util.reduce_model(ft, args.reduced_dimension)
        reduced = get_fasttext_embeddings(ft, words, args.reduced_dimension)
    
    elif args.method == 'custom':
        reduced = get_reduced_embeddings(args.reduced_embeddings_path, words, args.atomic)

    # compute loss
    target_sim = compute_similarity(targets)
    reduced_sim = compute_similarity(reduced)

    loss = torch.square(reduced_sim - target_sim).mean()
    print(f'Loss of reduced embeddings is: {loss.item():.4f}')


if __name__ == '__main__':
    main()
