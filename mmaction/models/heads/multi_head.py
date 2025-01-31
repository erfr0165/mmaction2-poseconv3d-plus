from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmaction.registry import MODELS
from mmaction.evaluation import top_k_accuracy
from mmaction.utils import ForwardResults, SampleList
from .i3d_head import I3DHead


@MODELS.register_module()
class MultiHead(nn.Module):
    def __init__(self,
                 head1,
                 head2):
        super(MultiHead, self).__init__()
        head1.pop('type')
        head2.pop('type')
        self.head1 = I3DHead(**head1)
        self.head2 = I3DHead(**head2)


    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores_head1 = self.head1(feats, **kwargs)
        cls_scores_head2 = self.head2(feats, **kwargs)

        return self.loss_by_feat(cls_scores_head1, cls_scores_head2, data_samples)


    def forward(self, x):
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs
    

    def loss_by_feat(self, cls_scores_head1: torch.Tensor, cls_scores_head2: torch.Tensor,
                     data_samples: SampleList) -> Dict:

        frame_dirs = [x.frame_dir for x in data_samples]
        data = [1 if 'data1' in x else 2 for x in frame_dirs]

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores_head1.device)
        labels = labels.squeeze()

        # seperate data1 and data2
        labels1, labels2, labels1_index = [], [], []
        try:
            for i, label in enumerate(labels):
                if data[i] == 1:
                    labels1.append(label)
                    labels1_index.append(i)
                else:
                    labels2.append(label)
        except TypeError:
            if data[0] == 1:
                labels1.append(torch.tensor(labels.item()))
                labels1_index.append(0)
            else:
                labels2.append(torch.tensor(labels.item()))
        labels1 = torch.stack(labels1) if len(labels1) > 0 else torch.tensor(labels1)
        labels1 = labels1.to(cls_scores_head1.device)

        labels2 = torch.stack(labels2) if len(labels2) > 0 else torch.tensor(labels2)
        labels2 = labels2.to(cls_scores_head1.device)

        mask = torch.zeros(cls_scores_head1.size(0), dtype=torch.bool)
        mask[labels1_index] = True

        # Use the mask to index the tensor
        scores1, scores2 = cls_scores_head1[mask], cls_scores_head2[~mask]

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.head1.num_classes \
                and cls_scores_head1.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        # compute accuracy
        # data1
        if  labels1.size()[0] != 0:
            top_k_acc1 = top_k_accuracy(scores1.detach().cpu().numpy(),
                                    labels1.detach().cpu().numpy(),
                                    self.head1.topk)
        else:  # no sample of data1 in batch
            top_k_acc1 = [0.0, 0.0]

        # data2
        if  labels2.size()[0] != 0:
            top_k_acc2 = top_k_accuracy(scores2.detach().cpu().numpy(),
                                    labels2.detach().cpu().numpy(),
                                    self.head2.topk)
        else:  # no sample of data2 in batch
            top_k_acc2 = [0.0, 0.0]

        # combine accuracy values
        for k, a, b in zip(self.head1.topk, top_k_acc1, top_k_acc2):
                r = (len(labels1) * a + len(labels2) * b) / len(labels)
                losses[f'top{k}_acc'] = torch.tensor(r, device=cls_scores_head1.device)

        # label smothing
        if self.head1.label_smooth_eps != 0:
            if labels1.numel() != 0:
                if cls_scores_head1.size() != labels1.size():
                    labels = F.one_hot(labels1, num_classes=self.head1.num_classes)
                labels1 = ((1 - self.head1.label_smooth_eps) * labels1 +
                        self.head1.label_smooth_eps / self.head1.num_classes)
        if self.head2.label_smooth_eps != 0:
            if labels2.numel() != 0:
                if cls_scores_head2.size() != labels2.size():
                    labels = F.one_hot(labels2, num_classes=self.head2.num_classes)
                labels2 = ((1 - self.head2.label_smooth_eps) * labels2 +
                        self.head2.label_smooth_eps / self.head2.num_classes)

        loss_cls = 0
        for scores, labels, head, cls_scores, labels_all in [(scores1, labels1, self.head1, cls_scores_head1, labels), 
                                         (scores2, labels2, self.head2, cls_scores_head2, labels)]:
            if labels.numel() != 0:
                loss_cls += head.loss_cls(scores, labels)
            else:
                pseudo_labels = labels_all.clone().fill_(1)
                dummy_loss = head.loss_cls(cls_scores, pseudo_labels).fill_(0.00001)
                loss_cls += dummy_loss

        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:

        cls_scores_head1 = self.head1(feats, **kwargs)
        cls_scores_head2 = self.head2(feats, **kwargs)
        return self.predict_by_feat(cls_scores_head1, cls_scores_head2, data_samples)
    
    def predict_by_feat(self, cls_scores_head1: torch.Tensor, cls_scores_head2: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        
        frame_dirs = [x.frame_dir for x in data_samples]
        data = [1 if 'data1' in x else 2 for x in frame_dirs]

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores_head1.device)
        labels = labels.squeeze()

        # seperate data1 and data2
        labels1, labels2, labels1_index = [], [], []
        try:
            for i, label in enumerate(labels):
                if data[i] == 1:
                    labels1.append(label)
                    labels1_index.append(i)
                else:
                    labels2.append(label)
        except TypeError:
            if data[0] == 1:
                labels1.append(torch.tensor(labels.item()))
                labels1_index.append(0)
            else:
                labels2.append(torch.tensor(labels.item()))
        labels1 = torch.stack(labels1) if len(labels1) > 0 else torch.tensor(labels1)
        labels1 = labels1.to(cls_scores_head1.device)

        labels2 = torch.stack(labels2) if len(labels2) > 0 else torch.tensor(labels2)
        labels2 = labels2.to(cls_scores_head1.device)

        mask = torch.zeros(cls_scores_head1.size(0), dtype=torch.bool)
        mask[labels1_index] = True

        num_segs = cls_scores_head1.shape[0] // len(data_samples)
        cls_scores_head1 = self.average_clip(cls_scores_head1, num_segs=num_segs)
        cls_scores_head2 = self.average_clip(cls_scores_head2, num_segs=num_segs)
        pred_labels_head1 = cls_scores_head1.argmax(dim=-1, keepdim=True).detach()
        pred_labels_head2 = cls_scores_head2.argmax(dim=-1, keepdim=True).detach()

        max_classes = max(self.head1.num_classes, self.head2.num_classes)

        for data_sample, score1, score2, pred_label_head1, pred_label_head2 in zip(data_samples, cls_scores_head1, cls_scores_head2,
                                                  pred_labels_head1, pred_labels_head2):
            if 'data1' in data_sample.frame_dir:
                if self.head1.num_classes < max_classes:
                    padding = max_classes - self.head1.num_classes
                    score1 = np.pad(score1.detach().cpu().numpy(), (0, padding), mode='constant', constant_values=0)
                score1 = score1 if isinstance(score1, torch.Tensor) else torch.from_numpy(score1)
                data_sample.set_pred_score(score1.to(cls_scores_head1.device))
                data_sample.set_pred_label(pred_label_head1)
            else:
                if self.head2.num_classes < max_classes:
                    padding = max_classes - self.head2.num_classes
                    score2 = np.pad(score2.detach().cpu().numpy(), (0, padding), mode='constant', constant_values=0)
                score2 = score2 if isinstance(score2, torch.Tensor) else torch.from_numpy(score2)
                data_sample.set_pred_score(score2.to(cls_scores_head2.device))
                data_sample.set_pred_label(pred_label_head2)
        return data_samples
    
    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:


        if self.head1.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.head1.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) +
                                     cls_scores.shape[1:])

        if self.head1.average_clips is None:
            return cls_scores
        elif self.head1.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.head1.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores