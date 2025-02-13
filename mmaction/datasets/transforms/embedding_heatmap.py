from typing import Dict, Tuple

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

import copy as cp
import pickle

import numpy as np
import time


@TRANSFORMS.register_module()
class GeneratePoseTargetWithEmbeddings(BaseTransform):
    """Generate pseudo heatmaps based on joint coordinates, object coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        (not working/not needed) with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.

        dimensions (int): Dimension of the word embeddings.
        embedding_path (str): Path to word embeddings that should be used.
        embedding_path2 (str): Path to word embeddings for second dataset.
        heatmaps (str): Options: ['sum', 'normalized_sum', 'weighted_normalization']. Determines the calculation for the heatmaps with word
            embeddings. Default: 'sum'.
        gaussian (str): Options: ['normal', 'one']. 'Normal' means causal gaussian calculation and 'one'
            means that the whole patch will be filled with ones instead of the gaussian values. Default: 'normal'.
        two_skeletons (bool): indicates that data with two different skeletons is processed and 
            causes the corresponding word embeddings to be loaded (embedding_path2 must be given). Default: False.
        use_confidences (bool): similar to use_score, needed when processing object data without using
            the score (when applying use_score=False "zero joints" will pe processed to, which results
            in longer training times). Default: True.
    """

    def __init__(self,
                 sigma: float = 0.6,
                 use_score: bool = True,
                 with_kp: bool = True,
                 with_limb: bool = False,
                 skeletons: Tuple[Tuple[int]] = ((0, 1), (0, 2), (1, 3),
                                                 (2, 4), (0, 5), (5, 7),
                                                 (7, 9), (0, 6), (6, 8),
                                                 (8, 10), (5, 11), (11, 13),
                                                 (13, 15), (6, 12), (12, 14),
                                                 (14, 16), (11, 12)),
                 double: bool = False,
                 left_kp: Tuple[int] = (),
                 right_kp: Tuple[int] = (),
                 left_limb: Tuple[int] = (),
                 right_limb: Tuple[int] = (),
                 scaling: float = 1.,
                 dimensions: int = 20,
                 embedding_path: str = None,
                 embedding_path2: str = None,
                 heatmaps: str = 'sum',
                 gaussian: str = 'normal',
                 two_skeletons: bool = False,
                 use_confidences: bool = True) -> None:

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp

        self.with_limb = False
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            '"with_kp" must be set as True')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling

        self.dimensions = dimensions
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        self.embeddings = embeddings
        self.heatmaps = heatmaps
        self.gaussian = gaussian
        self.two_skeletons = two_skeletons
        if self.two_skeletons:
            with open(embedding_path2, 'rb') as f:
                embeddings2 = pickle.load(f)
                self.embeddings2 = embeddings2
        self.use_confidences = use_confidences

    def generate_a_heatmap(self, arr: np.ndarray, centers: np.ndarray,
                           max_values: np.ndarray, frame_dir: str) -> None:
        """Generate pseudo heatmap for all dimensions in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: dimensions * img_h * img_w.
            centers (np.ndarray): The coordinates of all joints and objects
                (of multiple persons).
            max_values (np.ndarray): The max values of each joint/object.
        """

        sigma = self.sigma
        img_h = arr.shape[1]
        img_w = arr.shape[2]
    
        if self.heatmaps in {'normalized_sum', 'weighted_normalization'}:
            # for later heatmap computation
            arr_norm = arr.copy()

        # when processing 2 skeletons load embeddings for correct skeleton
        # here: determined by 'frame_dir' of the sample
        if self.two_skeletons:
            if 'data1' in frame_dir:
                embeddings = self.embeddings
            elif 'data2' in frame_dir:
                embeddings = self.embeddings2
        else:
            embeddings = self.embeddings

        # iteration through all keypoints from frame
        for center, max_value in zip(centers, max_values):
            for i, (c, m) in enumerate(zip(center, max_value)):
                
                if m < self.eps:  # skip keypoints with low score
                    continue

                mu_x, mu_y = c[0], c[1]
                st_x = max(int(mu_x - 3 * sigma), 0)
                ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
                st_y = max(int(mu_y - 3 * sigma), 0)
                ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
                x = np.arange(st_x, ed_x, 1, np.float32)
                y = np.arange(st_y, ed_y, 1, np.float32)

                # if the keypoint not in the heatmap coordinate system
                if not (len(x) and len(y)):
                    continue
                y = y[:, None]

                patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)

                # stack gaussian 'dimensions' times
                if self.gaussian == 'one':
                    patch_stack = np.ones((self.dimensions, len(y), len(x)), dtype=np.float32)
                else:
                    patch_stack = np.empty((self.dimensions,) + patch.shape, dtype=patch.dtype)
                    patch_stack[:] = patch

                # prepare word embedding values
                if self.use_confidences:
                    embs = embeddings[i][:, None, None] * m
                else:
                    embs = embeddings[i][:, None, None]
                # add word embeddings to heatmap
                np.add(arr[:, st_y:ed_y, st_x:ed_x], patch_stack * embs, out=arr[:, st_y:ed_y, st_x:ed_x])

                if self.heatmaps == 'normalized_sum':
                    arr_norm[:, st_y:ed_y, st_x:ed_x] += 1
                if self.heatmaps == 'weighted_normalization':
                    arr_norm[:, st_y:ed_y, st_x:ed_x] += patch_stack

        if self.heatmaps in {'normalized_sum', 'weighted_normalization'}:
            arr_norm = np.where(arr_norm==0, 1, arr_norm)

            if self.heatmaps == 'normalized_sum':
                arr /= arr_norm
            if self.heatmaps == 'weighted_normalization':
                arr = arr * (1 / arr_norm)

    def generate_heatmap(self, arr: np.ndarray, kps: np.ndarray,
                         max_values: np.ndarray, frame_dir: str) -> None:
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame.
                Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint.
                Shape: M * V.
        """

        self.generate_a_heatmap(arr, kps, max_values, frame_dir)



    def gen_an_aug(self, results: Dict) -> np.ndarray:
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            np.ndarray: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint'].astype(np.float32)
        kp_shape = all_kps.shape

        frame_dir = results['frame_dir']

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)

        ret = np.zeros([num_frame, self.dimensions, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else \
                np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores, frame_dir)

        return ret

    def transform(self, results: Dict) -> Dict:
        """Generate pseudo heatmaps based on joint coordinates and confidence.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        heatmap = self.gen_an_aug(results)

        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (
                self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        results[key] = heatmap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str