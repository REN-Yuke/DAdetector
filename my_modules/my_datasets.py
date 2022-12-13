import mmcv
from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class DADataset(CocoDataset):
    """Add domain labels for domain adaptation task,
    inherited from class CocoDataset.

    See :class:`mmdet.datasets.CocoDataset` for details.

    Args:
        domain_labels (str, list[str]) : Names of domain labels.
             Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 domain_labels=None,
                 **kargs):
        super(DADataset, self).__init__(ann_file=ann_file, pipeline=pipeline, **kargs)

        self.domain_labels = []
        if domain_labels is not None:
            if isinstance(domain_labels, str):
                domain_labels = [domain_labels]
            elif isinstance(domain_labels, list):
                assert mmcv.is_list_of(domain_labels, str)
            else:
                raise ValueError('domain_label must be either str or list of str')
            self.domain_labels = domain_labels

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (list[dict]): image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map and 'domain_labels'.
        """
        ann = super(DADataset, self)._parse_ann_info(img_info, ann_info)
        if self.domain_labels:
            ann['domain_labels'] = self.domain_labels
            for domain_label in self.domain_labels:
                ann[domain_label] = img_info[domain_label]
        return ann

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        super(DADataset, self).pre_pipeline(results)
        results['domain_fields'] = []

