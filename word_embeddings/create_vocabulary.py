import argparse
import fasttext
import fasttext.util
import nltk
import numpy as np
import torch

from nltk.corpus import wordnet as wn


# based on a list of starting words this script creates a larger vocabulary of words
# additionally it extract for these words the corresponding fasttext word embeddings
# and writes both (words + embeddings) to a .txt file


# possible starting words
IKEA_KEYPOINTS = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', \
             'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
IKEA_OBJECTS = ['table top', 'leg', 'shelf', 'side panel', 'front panel', 'bottom panel', 'rear panel']
ATTACH_KEYPOINTS = ['pelvis', 'spine navel', 'spine chest', 'neck', 'left clavicle', 'left shoulder', 'left elbow', \
					'left wrist', 'left hand', 'left hand tip', 'left thumb', 'right clavicle', 'right shoulder', 'right elbow', \
					'right wrist', 'right hand', 'right hand tip', 'right thumb', 'left hip', 'left knee', 'left ankle', 'left foot', \
					'right hip', 'right knee', 'right ankle', 'right foot', 'head', 'nose', 'left eye', 'left ear', 'right eye', 'right ear']
ATTACH_OBJECTS = ['screwdriver', 'cabinet foot', 'wall spacer top', 'screw no head', 'wrench', 'board', 'screw with head', 'threaded tupe female', \
				  'threaded rod', 'manual', 'hammer', 'wall spacer mounting']
SEPERATOR = ['_', '-']
TEST = ['eye', 'left-eye', 'right-eye', 'ear', 'nose', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'spine',
	'hammer', 'chisel', 'wrench', 'screwdriver', 'pliers', 'table', 'drawer', 'counter', 'board', 'panel', 'left', 'right']


def parse_args():
    parser = argparse.ArgumentParser(description='Parameter for vocabulary creation')
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=100,
        help='maximum number of words for the vocabulary')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./fasttext_embeddings/cc.en.300.bin',
        help='path to downloaded fasttext model')
    parser.add_argument(
        '--disable-add-atomics',
        action='store_true',
        default=False,
        help='if false words linked with e.g. hyphens will be added to vocabulary')
    parser.add_argument(
        '--disable-nouns-only',
        action='store_true',
        default=False,
        help='if false only nouns are added to vocabulary')
    parser.add_argument(
        '--custom-start-words-path',
        type=str,
        default='',
        help='path to .txt file with custom start words')

    args = parser.parse_args()
    return args


def get_related_words(words, ft_model, limit=1000, add_atomics=False, only_nouns=False):
    """Search for similar word to fill the word vocabulary.

        Args:
            words (list): List of starting words.
            ft_model (fasttextModel): Model to extract the embeddings from.
            limit (int): Maximum size of the vocabulary.
            add_atomics (bool): Example: 'left shoulder' is in starting words,
                if add_atomics=True it adds 'left' and 'shoulder' also to the vocabulary.
            only_nouns (bool): if True only nouns will be added to the vocabulary.

        Returns:
            list(torch.tensor): word embeddings.
            list: words.
    """
    dim = 300
    # remove dups + lower case
    synonyms = set([word.lower() for word in words])
    queue = list(synonyms)
    visited = synonyms
    embedded_words = []
    original_words = []
    # get embeddings for starting words
    original_words, embedded_words = get_embeddings(queue, embedded_words, original_words, ft_model, add_atomics)

    hypo = lambda s: s.hyponyms()
    hyper = lambda s: s.hypernyms()

    nltk.download('wordnet')
    # search similar words until vocabulary limit is reached
    while queue:
        current_word = queue.pop(0)
        for synset in wn.synsets(current_word): # synset = set of cognitive synonyms
            for lemma in synset.lemmas():
                if len(embedded_words) >= limit:
                    emb_tensor = torch.cat(embedded_words[:limit])
                    emb_tensor = emb_tensor.view(-1, dim)
                    original_words = original_words[:limit]
                    return emb_tensor, original_words
                
                # every superordinate word
                for hypernym in synset.closure(hyper, depth=3):
                    for hypernym_lemma in hypernym.lemmas():
                        hypernym_word = hypernym_lemma.name()
                        if hypernym_word not in visited:
                            visited.add(hypernym_word)
                            queue.append(hypernym_word)

                # every subordinate word
                for hyponym in synset.closure(hypo, depth=3):
                    for hyponym_lemma in hyponym.lemmas():
                        hyponym_word = hyponym_lemma.name()
                        if hyponym_word not in visited:
                            visited.add(hyponym_word)
                            queue.append(hyponym_word)

                # every part-of word
                for part_meronym in synset.part_meronyms():
                    for part_meronym_lemma in part_meronym.lemmas():
                        part_meronym_word = part_meronym_lemma.name()
                        if part_meronym_word not in visited:
                            visited.add(part_meronym_word)
                            queue.append(part_meronym_word)

                # only nouns
                if only_nouns:
                    pos = lemma.synset().pos()
                    if pos.startswith('n'):
                        word = lemma.name().lower()
                        if word not in synonyms:
                            synonyms.add(word)
                            get_embeddings([word], embedded_words, original_words, ft_model, add_atomics)
                else:
                    word = lemma.name().lower()
                    if word not in synonyms:
                        synonyms.add(word)
                        get_embeddings([word], embedded_words, original_words, ft_model, add_atomics)

    print(f'Less words than {limit} found!')
    emb_tensor = torch.cat(embedded_words)
    emb_tensor = emb_tensor.view(-1, dim)
    return emb_tensor, original_words


def get_embeddings(words, embedded_words, original_words, ft_model, add_atomics=False):
    """Collects the embeddings for all words.

        Args:
            words (list): List of words which embeddings should get collected.
            embedded_words (list): List of all word embeddings.
            original_words (list): List of all words that already got embedded.
            ft_model (fasttextModel): Model to extract the embeddings from.
            add_atomics (bool): Example: 'left shoulder' is in starting words,
                if add_atomics=True it adds 'left' and 'shoulder' also to the vocabulary.

        Returns:
            list: words.
            torch.tensor: word embeddings.
    """
    atomics = set()
    dim = 300
    for entry in words:
        if entry in ['_', '-']:
            embedded_words.append(torch.tensor(ft_model.get_word_vector(entry[0])))
            original_words.append(entry)
        else:
            if entry not in original_words:
                entry_ = entry.replace('_', ' ').replace('-', ' ').replace(':', ' ').replace('(', '').replace(')', '').split()
                # if entry consists of multiple words, take average of sum of embeddings
                if len(entry_) > 1:
                    avg_embedding = np.zeros((dim,), dtype=np.float32)
                    for e in entry_:
                        avg_embedding += ft_model.get_word_vector(e)
                        if add_atomics and e not in words:
                            atomics.add(e)
                    avg_embedding /= len(entry_)
                    embedded_words.append(torch.tensor(avg_embedding))
                    original_words.append(entry)
                else:
                    embedded_words.append(torch.tensor(ft_model.get_word_vector(entry_[0])))
                    original_words.append(entry)
    
    # get embeddings for atomic words
    for word in atomics:
        if word not in original_words:
            embedded_words.append(torch.tensor(ft_model.get_word_vector(word)))
            original_words.append(word)

    return original_words, embedded_words


def main():
    args = parse_args()

    ft = fasttext.load_model(args.model_path)
    print('fasttext model loaded succesfully')

    # define start words
    if args.custom_start_words_path:
        start_words = []
        with open(args.custom_start_words_path, 'r') as file:
            for line in file:
                start_words.append(line.strip())
    else:
        start_words = IKEA_KEYPOINTS + SEPERATOR + TEST + ATTACH_KEYPOINTS + IKEA_OBJECTS + ATTACH_OBJECTS
    # eliminate duplicates
    start_words = list(set(start_words))

    atomics = not args.disable_add_atomics
    nouns = not args.disable_nouns_only
    # collect similar words to fill the vocabulary
    embeddings, words = get_related_words(start_words, ft, limit=args.vocab_size, add_atomics=atomics, only_nouns=nouns)

    dictionary = {}
    for vec, word in zip(embeddings, words):
        dictionary[word] = np.asarray(vec)

    # write words + embeddings to .txt file
    with open("./vocabulary.txt", 'w', encoding="utf-8") as f:
        for item in dictionary.items():
            f.write("%s\t" % item[0])
            for t in item[1]:
                f.write("%s\t" % str(t))
            f.write("\n")


if __name__ == '__main__':
    main()