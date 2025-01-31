import argparse
import numpy as np
import os
import pickle


# this script will write the word embeddings of a given list of words (e.g. joints) to a .pkl file
# the order of the embeddings will be the order of the given words


IKEA_JOINTS = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', \
             'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

IKEA_OBJECTS = ['table top', 'leg', 'shelf', 'side panel', 'front panel', 'bottom panel', 'rear panel']

ATTACH_JOINTS = ['pelvis', 'spine navel', 'spine chest', 'neck', 'left clavicle', 'left shoulder', 'left elbow', \
					'left wrist', 'left hand', 'left hand tip', 'left thumb', 'right clavicle', 'right shoulder', 'right elbow', \
					'right wrist', 'right hand', 'right hand tip', 'right thumb', 'left hip', 'left knee', 'left ankle', 'left foot', \
					'right hip', 'right knee', 'right ankle', 'right foot', 'head', 'nose', 'left eye', 'left ear', 'right eye', 'right ear']

ATTACH_OBJECTS = ['screwdriver', 'cabinet foot', 'wall spacer top', 'screw no head', 'wrench', 'board', 'screw with head', 'threaded tupe female', \
				  'threaded rod', 'manual', 'hammer', 'wall spacer mounting']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='ikea',
        choices=['attach', 'ikea'],
        help='dataset for which the embeddings of the keypoints should be saved')
    parser.add_argument(
        '--objects',
        action='store_true',
        default=False,
        help='whether or not the objects embeddings should get saved too')
    parser.add_argument(
        '--output-file-path',
        type=str,
        default='./embedding.pkl',
        help='file to save embeddings in')
    parser.add_argument(
        '--reduced-embeddings-file',
        type=str,
        default='./embeddings.txt',
        help='input file with reduced word embeddings')
    parser.add_argument(
        '--combination-method',
        type=str,
        default='space',
        choices=['space', 'hyphen', 'underline'],
        help='combination method for embeddings of keypoints that consist of multiple words')
    parser.add_argument(
        '--normalization',
        action='store_true',
        default=False,
        help='normalization of word embeddings')
    parser.add_argument(
        '--vector-length',
        type=float,
        default=1.0,
        help='length to which the word embeddings should be normalized')
    parser.add_argument(
        '--disable-atomic',
        action='store_true',
        default=False,
        help='Example for term "left shoulder": if False it will add the embeddings from \
              "left" and "shoulder" and divide this by two; if True it will simply look for \
              "left shoulder" in the .txt file and take that embedding')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read embedding from input file
    word_to_embedding = {}
    with open(args.reduced_embeddings_file, 'r', encoding='utf-8') as f:
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

    # get keypoint names
    joints = globals()[f'{args.dataset.upper()}_JOINTS']
    objects = []
    if args.objects:
        objects = globals()[f'{args.dataset.upper()}_OBJECTS']
    keypoints = joints + objects

    # order of words is identical to order of keypoints in dataset
    embeddings = []
    emb = next(iter(word_to_embedding.items()))
    size = len(emb[1])
    # get embeddings
    for keypoint in keypoints:
        embedding = np.zeros(size)
        if args.disable_atomic:
            try:
                embedding = word_to_embedding[keypoint]
                embeddings.append(embedding)
            except KeyError:
                raise KeyError(f'The word "{word}" does not appear in {args.reduced_embeddings_file}!')
        else:
            try:
                keypoint_ = keypoint.replace('-', ' ').replace('_', ' ').split(' ')
                for word in keypoint_:
                    embedding += word_to_embedding[word]
                if args.combination_method == 'space':
                    embedding /= len(keypoint_)
                elif args.combination_method == 'hyphen':
                    embedding += (word_to_embedding['-'] * (len(keypoint_) - 1))
                    embedding /= (len(keypoint_) * 2 - 1)
                elif args.combination_method == 'underline':
                    embedding += (word_to_embedding['_'] * (len(keypoint_) - 1))
                    embedding /= (len(keypoint_) * 2 - 1)
                embeddings.append(embedding)
            except KeyError:
                raise KeyError(f'The word "{word}" does not appear in {args.reduced_embeddings_file}!')

    # normalize embeddings
    if args.normalization:
        for i, embedding in enumerate(embeddings):
            norm = np.linalg.norm(embedding)
            if norm == 0:
                embeddings[i] = np.zeros_like(embedding)  
            else:
                embeddings[i] = (embedding / norm) * args.vector_length

    # write embeddings to output file
    directory = os.path.dirname(args.output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(args.output_file_path, 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    main()
