from cProfile import label
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer

import torch.nn.functional as F

@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Recognizer3D, self).__init__(backbone, cls_head, neck, train_cfg, test_cfg)
        # self.momentum_score = torch.zeros((1, 17))
        # self.momentum_alpha = 0.9
        # self.pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        # self.clip_fc = nn.Linear(16, 32)
        # self.cas_fc = nn.Linear(1536, 17)
        # self.class_fc = nn.Linear(1536, 1)
        # self.output_record = []

    def wsal_pred(self, x):
        feature_x = self.pool_2d(x) # (N, 1536, 16, 1, 1)
        feature_x = feature_x.squeeze(-1).squeeze(-1) # (N, 1536, 16)
        feature_x = self.clip_fc(feature_x) # (N, 1536, 32) (T, D)
        cas = self.cas_fc(feature_x.transpose(1, 2)).transpose(1, 2) # (N, 17, 32)
        class_vector = self.class_fc(feature_x.transpose(1, 2)) # (N, 32, 1)
        class_score = (cas @ class_vector).transpose(1, 2) # (N, 1, 17)
        class_score = class_score.squeeze(1) # (N, 17)

        return class_score, cas

    def class_frame(self, cas, imgs, index, sample_flag=True):
        frame_list = cas[:, index].squeeze()
        init_thr = 0.
        frame_index = (frame_list > init_thr)
        num_frame = frame_index.sum()
        # print(num_frame, end=' ')
        while frame_index.sum() == 0:
            init_thr -= 0.1
            frame_index = (frame_list > init_thr)
        frame_img = imgs[:, :, frame_index, :, :]
        # print(frame_index.sum())
        
        # upsample_util = nn.Upsample(scale_factor=(all_frame_num / num_frame, 1, 1), mode='trilinear')
        # up_sample_imgs = upsample_util(frame_img)
        up_sample_imgs = F.interpolate(frame_img, imgs.shape[-3:], mode='trilinear')
        assert up_sample_imgs.shape == imgs.shape
        x = self.extract_feat(up_sample_imgs)
        cls_score = self.cls_head(x)
        # single_label = F.one_hot(index, num_classes=17)
        single_label = torch.tensor([1]) if sample_flag else torch.tensor([0])
        single_label = single_label.to(cls_score.device)
        backup_kwargs = {}
        loss_cls = self.cls_head.loss(cls_score[:, index], single_label.detach(), **backup_kwargs)

        return loss_cls

    def wsal_pred_label(self, x, imgs, labels, weight=1e-2):
        all_frame_num = imgs.shape[2]
        class_score, cas = self.wsal_pred(x)
        # index_list = torch.where(class_score > 0)[1]
        index_list = torch.where(labels == 1)[1]
        
        single_label_loss = 0
        length = 0
        for i in range(len(index_list) - 1, -1, -1):
            index = index_list[i]
            # print(index)
            # if labels[0, index] != 1:
            #     continue
            length += 1
            ''' for CUDA out of memory '''
            if length == 2:
                break

            loss_cls = self.class_frame(cas, imgs, index, sample_flag=True)
            single_label_loss += loss_cls['loss_cls']
        # if length == 0:
        #     return -1

        sort_index = torch.argsort(class_score).squeeze()
        count = 0
        for i in range(len(sort_index) - 1, -1, -1):
            if sort_index[i] in index_list or count == 2:
                continue
            count += 1
            length += 1
            index = sort_index[i]
            if count == 2:
                break

            loss_cls = self.class_frame(cas, imgs, index, sample_flag=False)
            single_label_loss += loss_cls['loss_cls']

        return single_label_loss * weight / length
            
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # (1, 3, 32, 224, 224)
        losses = dict()

        x = self.extract_feat(imgs) # (1, 1536, 16, 7, 7)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        ''' Weakly-supervise action location'''
        # class_score = self.wsal_pred(x)[0]
        # import torch.nn.functional as F
        # wsal_loss_cls = F.binary_cross_entropy(F.sigmoid(class_score), labels)
        # losses.update({"wsal_loss" : wsal_loss_cls})

        # ''' Weakly-supervise action location single frame label '''
        # single_label_loss = self.wsal_pred_label(x, imgs, labels, weight=5e-2)
        # # if single_label_loss != -1:
        # losses.update({"single_label_loss" : single_label_loss})

        ''' GCN '''
        # feature = self.cls_head.avg_pool(x)
        # feature = self.cls_head.dropout(feature)
        # feature = feature.view(feature.shape[0], -1) # (1, 1024)
        # feature = self.gcn_head(feature) # (1, 2048)

        # classes_feature = self.gcn_net() # (2048, 17)
        # cls_score = torch.matmul(feature, classes_feature) # (1, 17)
        # cls_score = self.fc_cls(cls_score) # (1, 1024)
        # cls_score = self.cls_head.fc_cls(cls_score) # (1, 17)

        ''' origin classification head '''
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()

        start_epoch = 0
        ''' momentu logits'''
        # if kwargs['epoch'] >= start_epoch:
        #     import torch.nn.functional as F
        #     if self.momentum_score.device != cls_score.device:
        #         self.momentum_score = self.momentum_score.to(cls_score.device)
        #     self.dist_gather_momentum_scores(cls_score, gt_labels, kwargs['epoch'])
        #     gt_index = (gt_labels.long() == 1)
        # if kwargs['epoch'] >= start_epoch + 3:
        #     self.momentum_alpha = 0.99
        #     kd_loss = F.mse_loss(cls_score[:, gt_index], self.momentum_score[:, gt_index].detach())
        #     losses.update({'kd_loss' : kd_loss * 1e-2})

        backup_kwargs = {}
        # loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **backup_kwargs)
        # print('-' * 30, loss_cls, gt_labels)
        losses.update(loss_cls)
        # self.output_record.append(
        #     {
        #         'frame_dir' : kwargs['frame_dir'],
        #         'cls_loss' : loss_cls['loss_cls'].item(),
        #         'cls_score' : cls_score.detach().clone().squeeze().cpu().numpy().tolist(),
        #         'gt_label' : gt_labels.detach().clone().squeeze().cpu().numpy().tolist(),
        #     }
        # )

        return losses

    def dist_gather_momentum_scores(self, cls_score, gt_labels, epoch):
        import torch.nn.functional as F
        import torch.distributed as dist

        current_rank = dist.get_rank()
        world_size = dist.get_world_size()
        scores_gather_list = [torch.zeros_like(cls_score) for _ in range(world_size)]
        gt_label_gather_list = [torch.zeros_like(gt_labels) for _ in range(world_size)]
        dist.all_gather(scores_gather_list, cls_score, async_op=False)
        dist.all_gather(gt_label_gather_list, gt_labels, async_op=False)                
        
        for j in range(world_size):
            gt_index = (gt_label_gather_list[j].long() == 1)
            # if current_rank == 0:
                # print(scores_gather_list[j][0, gt_index])
            self.momentum_score = (1 - gt_label_gather_list[j]) * self.momentum_score \
                                    + (self.momentum_alpha) * (self.momentum_score * gt_index) \
                                    + (1 - self.momentum_alpha) * scores_gather_list[j] * gt_index
        if current_rank == 0:
            # print(scores_gather_list[0][0, gt_index])
            # print(self.momentum_score)
            pass

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        ''' origin classification loss '''
        cls_score = self.cls_head(feat)
        # -------------------------------------------------------------------
        logits_score = cls_score.mean(dim=0, keepdim=True)

        ''' weakly-supervise action localtion '''
        # wsal_cls_score = self.wsal_pred(feat)[0]
        # wsal_logits_score = wsal_cls_score.mean(dim=0, keepdim=True)
        # wsal_cls_score = self.average_clip(wsal_cls_score, num_segs)
        # -------------------------------------------------------------------

        cls_score = self.average_clip(cls_score, num_segs)
        # print(logits_score.squeeze())
        return cls_score, logits_score
        # return cls_score, logits_score, wsal_cls_score, wsal_logits_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        # ---------------------------------------------------------------
        # return self._do_test(imgs).cpu().numpy()
        # ---------------------------------------------------------------
        return self._do_test(imgs)

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)



# ==================================================================================================

# ==================================================================================================