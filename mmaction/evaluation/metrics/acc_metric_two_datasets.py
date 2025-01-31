# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (get_weighted_score, mean_average_precision,
                                 mmit_mean_average_precision, confusion_matrix)
from mmaction.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    labels1, labels2, scores1, scores2 = [], [], [], []

    # seperate data
    for label, score in zip(labels, scores):
        zero_count = np.sum(score == 0)
        if zero_count >= 3:
            labels1.append(label)
            scores1.append(score)
        else:
            labels2.append(label)
            scores2.append(score)
    
    mean_class_acc1 = mean_class_acc2 = 0
    # for labels, scores, mean_class_acc_var in [(labels1, scores1, 'mean_class_acc1'), (labels2, scores2, 'mean_class_acc2')]:
    #     if len(labels) > 0:
    #         pred = np.argmax(scores, axis=1)
    #         cf_mat = confusion_matrix(pred, labels).astype(float)

    #         cls_cnt = cf_mat.sum(axis=1)
    #         cls_hit = np.diag(cf_mat)

    #         mean_class_acc = np.mean([hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])
    #         accs = [hit / cnt for cnt, hit in zip(cls_cnt, cls_hit)]  # Compute accuracies

    #         # Assign mean_class_acc to the appropriate variable
    #         if mean_class_acc_var == 'mean_class_acc1':
    #             mean_class_acc1 = mean_class_acc
    #         else:
    #             mean_class_acc2 = mean_class_acc
    # dataset 1
    if len(labels1) > 0:
        pred1 = np.argmax(scores1, axis=1)
        cf_mat1 = confusion_matrix(pred1, labels1).astype(float)

        cls_cnt1 = cf_mat1.sum(axis=1)
        cls_hit1 = np.diag(cf_mat1)

        mean_class_acc1 = np.mean(
            [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt1, cls_hit1)])
        
        accs = []
        for cnt, hit in zip(cls_cnt1, cls_hit1):
            acc = hit / cnt
            accs.append(acc)

    # dataset 2
    if len(labels2) > 0:
        pred2 = np.argmax(scores2, axis=1)
        cf_mat2 = confusion_matrix(pred2, labels2).astype(float)

        cls_cnt2 = cf_mat2.sum(axis=1)
        cls_hit2 = np.diag(cf_mat2)

        mean_class_acc2 = np.mean(
            [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt2, cls_hit2)])
        
        accs = []
        for cnt, hit in zip(cls_cnt2, cls_hit2):
            acc = hit / cnt
            accs.append(acc)

    mean_class_acc = (len(labels1) * mean_class_acc1 + len(labels2) * mean_class_acc2) / len(labels)

    return mean_class_acc, mean_class_acc1, mean_class_acc2


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    labels1, labels2, scores1, scores2 = [], [], [], []

    # seperate data
    for label, score in zip(labels, scores):
        zero_count = np.sum(score == 0)
        if zero_count >= 3:
            labels1.append(label)
            scores1.append(score)
        else:
            labels2.append(label)
            scores2.append(score)
    
    res = []
    labels = np.array(labels)[:, np.newaxis]
    labels1 = np.array(labels1)[:, np.newaxis]
    labels2 = np.array(labels2)[:, np.newaxis]
    data1, data2 = [], []

    for k in topk:
        # all
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

        # data 1
        if len(labels1) > 0:
            max_k_preds = np.argsort(scores1, axis=1)[:, -k:][:, ::-1]
            match_array = np.logical_or.reduce(max_k_preds == labels1, axis=1)
            topk_acc_score = match_array.sum() / match_array.shape[0]
            data1.append(topk_acc_score)
        # data 2
        if len(labels2) > 0:
            max_k_preds = np.argsort(scores2, axis=1)[:, -k:][:, ::-1]
            match_array = np.logical_or.reduce(max_k_preds == labels2, axis=1)
            topk_acc_score = match_array.sum() / match_array.shape[0]
            data2.append(topk_acc_score)
    
    if len(labels1) == 0:
        data1 = [0, 0]
    
    if len(labels2) == 0:
        data2 = [0, 0]

    return res, data1, data2


@METRICS.register_module()
class AccMetric2(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = self.calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc, top_data1, top_data2 = top_k_accuracy(preds, labels, topk)
                for k, acc, d1, d2 in zip(topk, top_k_acc, top_data1, top_data2):
                    eval_results[f'top{k}'] = acc
                    eval_results[f'data1_top{k}'] = d1
                    eval_results[f'data2_top{k}'] = d2

            if metric == 'mean_class_accuracy':
                mean1, mean_data1, mean_data2 = mean_class_accuracy(preds, labels)
                eval_results['mean1'] = mean1
                eval_results['mean_data1'] = mean_data1
                eval_results['mean_data2'] = mean_data2

            if metric in [
                    'mean_average_precision',
                    'mmit_mean_average_precision',
            ]:
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(preds, labels)
                    eval_results['mean_average_precision'] = mAP

                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(preds, labels)
                    eval_results['mmit_mean_average_precision'] = mAP

        return eval_results