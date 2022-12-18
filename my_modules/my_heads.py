from abc import ABCMeta

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import (HEADS, RPNHead, StandardRoIHead, Shared2FCBBoxHead,
                          build_head, build_loss)
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import images_to_levels, multi_apply, bbox2roi

from my_modules.my_layers import RevGrad


@HEADS.register_module()
class DARPNHead(RPNHead):
    """
    RPN head can be implemented for 2D detection domain adaptation task.

    See :class:`mmdet.models.dense_heads.RPNHead` for details.
    """
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      with_domain=False,
                      domain_label=None,
                      **kwargs):
        """
        See :class:`mmdet.models.dense_heads.BaseDenseHead.forward_train` for details.

        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class RPNHead.
            domain_label: (Tensor) Image-level domain label for source.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)  # outs = self.forward(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, None, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                           with_domain=with_domain, domain_label=domain_label)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             with_domain=False,
             domain_label=None):
        """Compute losses of the head.

        See :class:`mmdet.models.dense_heads.AnchorHead.loss` for details.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class RPNHead.
            domain_label: (Tensor) Image-level domain label for source.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            with_domain=with_domain,
            domain_label=domain_label)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False,
                    with_domain=False,
                    domain_label=None):
        """Compute regression and classification targets for anchors in
        multiple images.

        See :class:`mmdet.models.dense_heads.AnchorHead.get_targets` for details.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class RPNHead.
            domain_label: (Tensor) Image-level domain label for source.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]

        # if with_domain==True, the weights for classification loss and regression loss
        # should be zero, which means data from target source doesn't help train RPN head.
        # It can be set here, or can also be set in self._get_targets_single method.
        if with_domain:
            assert len(domain_label) == len(all_label_weights) == len(all_bbox_weights), \
                'all images should have domain label'
            for i, l in enumerate(domain_label):
                # l == 1 means target source
                if l == 1:
                    all_label_weights[i] = torch.zeros_like(all_label_weights[i])
                    all_bbox_weights[i] = torch.zeros_like(all_bbox_weights[i])
                # l == 0 means source source
                elif l == 0:
                    continue
                else:
                    raise ValueError('domain label must be 0 or 1, but it is {}.'.format(l))

        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


@HEADS.register_module()
class DAStandardRoIHead(StandardRoIHead):
    """
    RoI head can be implemented for 2D detection domain adaptation task.

    See :class:`mmdet.models.roi_heads.StandardRoIHead` for details.
    """
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      with_domain=False,
                      domain_label=None,
                      **kwargs):
        """
        See :class:`mmdet.models.roi_heads.StandardRoIHead.forward_train` for details.

        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class StandardRoIHead.
            domain_label: (Tensor) Image-level domain label for source.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas,
                                                    with_domain=with_domain,
                                                    domain_label=domain_label)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        if self.with_bbox:
            roi_dict = dict(batch_idx=bbox_results['batch_idx'],
                            bbox_feats=bbox_results['bbox_feats'])
            return losses, roi_dict
        else:
            return losses, None

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, with_domain=False, domain_label=None):
        """
        Run forward function and calculate loss for box head in training.

        See :class:`mmdet.models.roi_heads.StandardRoIHead._bbox_forward_train` for details.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class StandardRoIHead.
            domain_label: (Tensor) Image-level domain label for source.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg,
                                                  with_domain=with_domain,
                                                  domain_label=domain_label)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        # get batch_idx from rois to indicate every bbox_feats belongs to which image
        batch_idx = torch.tensor([r[0] for r in rois], device='cuda')

        # add bbox_results.batch_idx
        bbox_results.update(loss_bbox=loss_bbox, batch_idx=batch_idx)
        return bbox_results


@HEADS.register_module()
class DAShared2FCBBoxHead(Shared2FCBBoxHead):
    """
    Modified bounding box head for domain adaptation.
    """
    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    with_domain=False,
                    domain_label=None):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        See :class:`mmdet.models.roi_heads.bbox_heads.BBoxHead.get_targets` for details.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

            with_domain (bool): If with_domain == True, the loss calculated by the images with
                specific domain_label will multiply 0. That is to say, it will be deprecated.
                Defaults to False, which means this module work as same as class StandardRoIHead.
            domain_label: (Tensor) Image-level domain label for source.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        # if with_domain==True, the weights for classification loss and regression loss
        # should be zero, which means data from target source doesn't help train RoI head.
        # It can be set here, or can also be set in self._get_targets_single method.
        if with_domain:
            assert len(domain_label) == len(label_weights) == len(bbox_weights), \
                'all images should have domain label'
            for i, l in enumerate(domain_label):
                # l == 1 means target source
                if l == 1:
                    label_weights[i] = torch.zeros_like(label_weights[i])
                    bbox_weights[i] = torch.zeros_like(bbox_weights[i])
                # l == 0 means source source
                elif l == 0:
                    continue
                else:
                    raise ValueError('domain label must be 0 or 1, but it is {}.'.format(l))

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


@HEADS.register_module()
class DAImgHead(BaseModule, metaclass=ABCMeta):
    """
    Domain classification head for image level(whole feature map level)
    """
    # TODO: 后续可以修改为一个类似 Mask-RCNN 的检测头
    def __init__(self,
                 in_channels=256,
                 num_classes=1,
                 loss_img_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=True, loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        """
        Initiate domain classification head for image level.

        :param num_classes: number of domain label. Defaults to 1.
        :param in_channels: input channel number of each level of whole feature map.
        :param init_cfg: initiation config.
        :param args:
        :param kwargs:
        """
        super(DAImgHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0 or type(self.num_classes) != type(0):
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc1 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.in_channels,
                             kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.num_classes,
                             kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.loss_img_cls = build_loss(loss_img_cls)

    def forward(self, x, gt_label, with_consistency=False, **kwargs):
        """
        Forward pass the domain head in train mode, this class has no forward_test function.

        :param x: (Tensor or List[Tensor]) the whole feature map,
                    every Tensor has shape (N, C, H, W), where N is batch size,
                    C is equal to self.in_channels, H and W may be different.
        :param gt_label: (Tensor), here it is image-level domain label with shape = (N,).
        :param with_consistency: (bool), whether evaluate consistency of both domain head.
        :param kwargs:
        :return: (Tensor), domain classification loss for image level.
        """
        if isinstance(x, torch.Tensor):
            assert x.ndim == 4, 'single level feature map should have 4 dimensions'
        elif isinstance(x, tuple):
            assert mmcv.is_tuple_of(x, torch.Tensor)
            assert [i.ndim == 4 for i in x], 'feature map of every level should have 4 dimensions'
        else:
            raise ValueError('feature map x must be either Tensor or tuple of Tensor')

        assert all(len(i) == len(gt_label) for i in x)

        # forward pass
        cls_logit = []
        for i in x:
            i = self.fc1(i)
            i = F.relu(i)
            i = self.fc2(i)
            i = F.relu(i)
            i = self.global_pool(i)
            # cls_logit: list[Tensor]
            # Tensor: (batch size, self.num_classes, 1, 1) -> (batch size, self.num_classes)
            cls_logit.append(i.reshape(i.shape[0], i.shape[1]))

        # change the result from List[Tensor] to Tensor format
        # domain_img_cls_logit: Tensor: (batch size, self.num_classes, len(x))
        domain_img_cls_logit = torch.stack(cls_logit, dim=-1)

        # broadcast gt_label to match with prediction shape
        domain_img_gt_label = gt_label.reshape(-1, 1, 1).expand_as(domain_img_cls_logit)

        # calculate loss for DAImgHead
        loss_img_cls = self.loss_img_cls(domain_img_cls_logit, domain_img_gt_label, **kwargs)

        if with_consistency:
            return dict(loss_img_cls=loss_img_cls), domain_img_cls_logit
        else:
            return dict(loss_img_cls=loss_img_cls), None


@HEADS.register_module()
class DAInsHead(BaseModule, metaclass=ABCMeta):
    """
    Domain classification head for instance level(RoI level)
    """
    def __init__(self,
                 in_channels=256,
                 num_classes=1,
                 loss_ins_cls=dict(type='FocalLoss', gamma=5.0, alpha=0.5, loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        """
        Initiate domain classification head for instance level

        :param num_classes: number of domain label. Defaults to 1.
        :param in_channels: input channel number of each level of whole feature map.
        :param init_cfg: initiation config.
        :param args:
        :param kwargs:
        """
        super(DAInsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0 or type(self.num_classes) != type(0):
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc1 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.in_channels*2,
                             kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=self.in_channels*2,
                             out_channels=self.in_channels*2,
                             kernel_size=1)
        self.fc3 = nn.Conv2d(in_channels=self.in_channels*2,
                             out_channels=self.num_classes,
                             kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.loss_ins_cls = build_loss(loss_ins_cls)

    def forward(self, roi_feats, gt_label, with_consistency=False, **kwargs):
        """
        Forward pass the domain head in train mode, this class has no forward_test function.

        :param roi_feats: RoI's feature map, with shape (num_rois, C, H, W).
        :param gt_label: (Tensor), here it is instance-level domain label with shape (num_rois,).
        :param with_consistency: (bool), whether evaluate consistency of both domain head.
        :param kwargs:
        :return: (Tensor), domain classification loss for image level.
        """
        if isinstance(roi_feats, torch.Tensor):
            assert roi_feats.ndim == 4, 'single level feature map should have 4 dimensions'
        else:
            raise ValueError("RoI's feature map roi_feats must be either Tensor or list of Tensor")

        assert len(roi_feats) == len(gt_label)

        # forward pass
        roi_feats = F.relu(self.fc1(roi_feats))
        roi_feats = F.dropout(roi_feats, p=0.5)
        roi_feats = F.relu(self.fc2(roi_feats))
        roi_feats = F.dropout(roi_feats, p=0.5)
        roi_feats = self.fc3(roi_feats)
        roi_feats = self.global_pool(roi_feats)

        # For class mmdet.models.losses.FocalLoss:
        # domain_ins_cls_logit: Tensor: (num_rois, self.num_classes, 1, 1) -> (num_rois, self.num_classes)
        domain_ins_cls_logit = torch.reshape(roi_feats, (roi_feats.shape[0], roi_feats.shape[1]))

        # For class mmdet.models.losses.FocalLoss:
        # domain_ins_gt_label: Tensor: (num_rois, )
        domain_ins_gt_label = gt_label

        # calculate loss for DAInsHead
        loss_ins_cls = self.loss_ins_cls(domain_ins_cls_logit, domain_ins_gt_label, **kwargs)

        if with_consistency:
            return dict(loss_ins_cls=loss_ins_cls), domain_ins_cls_logit
        else:
            return dict(loss_ins_cls=loss_ins_cls), None


@HEADS.register_module()
class DAHead(BaseModule, metaclass=ABCMeta):
    """
    Use two domain classification heads (image-level, instance-level) and
    calculate consistency between both to solve domain adaptive task.
    """
    def __init__(self,
                 alpha=1.0,
                 domain_image_head=None,
                 domain_instance_head=None,
                 with_consistency=False,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        """
        :param alpha: (float) hyperparameter in GRL, no effect in the forward pass,
                        the gradient multiplies negative alpha in the backward pass.
        :param domain_image_head: image-level domain classification head.
        :param domain_instance_head: instance-level domain classification head.
        :param with_consistency: the consistency of the both domain head,
                        only can be True when have both heads.
        """
        super(DAHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.alpha = alpha
        # Gradient Reverse Layer
        self.grl = RevGrad(alpha=self.alpha)

        if domain_image_head is not None:
            self.domain_image_head = build_head(domain_image_head)
        if domain_instance_head is not None:
            self.domain_instance_head = build_head(domain_instance_head)
        if domain_image_head is None and domain_instance_head is None:
            raise Exception('No domain head available !')
        if domain_image_head is not None and domain_instance_head is not None:
            self.with_consistency = with_consistency
        else:
            self.with_consistency = False

    @property
    def with_domain_image_head(self):
        """bool: whether the domain head contains an image-level domain classification head."""
        return hasattr(self, 'domain_image_head') and self.domain_image_head is not None

    @property
    def with_domain_instance_head(self):
        """bool: whether the domain head contains an instance-level domain classification head."""
        return hasattr(self, 'domain_instance_head') and self.domain_instance_head is not None

    def forward(self, x, roi_dict, domain_label):
        """
        Forward pass the image-level and instance-level domain head in train mode,
        this class has no forward_test function.

        :param x: (Tensor or tuple[Tensor]) Feature map from neck (if it has) or backbone.
        :param roi_dict: (dict[str, Tensor]) A dictionary of RoI's feature map.
        :param domain_label: (Tensor) Image-level domain label for source.
        :return: dict[str, Tensor] A dictionary of loss components.
        """
        losses = {}

        # image-level domain cls head forward and loss
        if self.with_domain_image_head:
            # Gradient Reverse Layer
            x = self.grl(x)

            loss_img_cls, domain_img_cls_logit = self.domain_image_head(x, domain_label,
                                                                        self.with_consistency)
            losses.update(loss_img_cls)

        # instance-level domain cls head forward and loss
        if self.with_domain_instance_head:
            # get RoI's feature map from RCNN
            roi_feats = roi_dict['bbox_feats']
            # Gradient Reverse Layer
            roi_feats = self.grl(roi_feats)

            # Change image-level domain label to instance-level domain_label
            batch_idx = roi_dict['batch_idx']
            roi_domain_label = torch.tensor([domain_label[i.int()] for i in batch_idx], device='cuda')

            loss_ins_cls, domain_ins_cls_logit = self.domain_instance_head(roi_feats, roi_domain_label,
                                                                           self.with_consistency)
            losses.update(loss_ins_cls)

        # calculate consistency loss
        if self.with_consistency:
            loss_consistency = self.consistency(domain_img_cls_logit, domain_ins_cls_logit, batch_idx)
            losses.update(loss_consistency)

        return losses

    def consistency(self, domain_img_cls_logit, domain_ins_cls_logit, batch_idx):
        """
        Consistency check between two separated domain heads' results.

        :param domain_img_cls_logit: (Tensor), cls logit with shape (batch size, self.num_classes, len(x)).
        :param domain_ins_cls_logit: (Tensor), cls logit with shape (num_rois, self.num_classes).
        :param batch_idx: (Tensor), indicate every bbox_feats belongs to which image, has shape (num_rois,).
        :return: loss for consistency.
        """
        # Tensor: (batch size, self.num_classes, len(x)) -> (batch size, self.num_classes)
        pred_img = torch.mean(domain_img_cls_logit, dim=-1)
        # Tensor: (num_rois, self.num_classes) -> (batch size, self.num_classes)
        pred_ins = torch.zeros_like(pred_img)
        for i in batch_idx.int():
            pred_ins[i] += domain_ins_cls_logit[i]
        pred_ins = pred_ins / 512  # divided by model.train_cfg.rcnn.sampler.num (number of RoIs in each image)
        loss_consistency = F.mse_loss(pred_ins, pred_img)
        return dict(loss_consistency=loss_consistency)

