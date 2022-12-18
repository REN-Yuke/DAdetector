import mmcv
import torch

from mmdet.models import DETECTORS, build_head
from mmdet.models.detectors import TwoStageDetector


@DETECTORS.register_module()
class DATwoStageDetector(TwoStageDetector):
    """Two stage detector can be implemented for 2D detection domain adaptation task.

    See :class:`mmdet.models.detectors.TwoStageDetector` for details.

    Args:
        domain_labels (str, list[str]) : Names of domain labels.
             Default: None.
        domain_head (classification head) : classification heads using domain labels.
             Default: None.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 domain_labels=None,
                 domain_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DATwoStageDetector, self).__init__(backbone=backbone,
                                                 neck=neck,
                                                 rpn_head=rpn_head,
                                                 roi_head=roi_head,
                                                 train_cfg=train_cfg,
                                                 test_cfg=test_cfg,
                                                 pretrained=pretrained,
                                                 init_cfg=init_cfg)
        if domain_labels is not None:
            if isinstance(domain_labels, str):
                domain_labels = [domain_labels]
            elif isinstance(domain_labels, list):
                assert mmcv.is_list_of(domain_labels, str)
            else:
                raise ValueError('domain_label must be either str or list of str')
        self.domain_labels = domain_labels

        if domain_head is not None:
            self.domain_head = build_head(domain_head)

    @property
    def with_domain(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'domain_head') and self.domain_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        # domain_head
        # TODO: calculate domain_head flops
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        if self.with_domain:
            # select domain_label using in domain_head
            # only use the first domain label(i.e. self.domain_labels[0]) for now
            rain = kwargs[self.domain_labels[0]]
            assert isinstance(rain, torch.Tensor), 'domain label must be a single Tensor.'
        else:
            rain = None

        # RPN forward and loss
        if self.with_rpn:
            # configure for RPN proposal
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # calculate RPN loss and output RPN proposal bboxes and score(for containing an object)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                with_domain=self.with_domain,
                domain_label=rain,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RoI forward and loss
        roi_losses, roi_dict = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                           gt_bboxes, gt_labels,
                                                           gt_bboxes_ignore, gt_masks,
                                                           with_domain=self.with_domain,
                                                           domain_label=rain,
                                                           **kwargs)
        losses.update(roi_losses)

        if self.with_domain:
            # domain head forward and loss
            domain_loss = self.domain_head(x, roi_dict=roi_dict, domain_label=rain)
            losses.update(domain_loss)

        return losses
