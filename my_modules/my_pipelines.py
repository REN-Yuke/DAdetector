import mmcv
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, Collect


@PIPELINES.register_module()
class DALoadAnnotations(LoadAnnotations):
    """Add domain labels for domain adaptation task,
    inherited from class LoadAnnotations.

    See :class:`mmdet.datasets.pipelines.LoadAnnotations` for details.

    Args:
        domain_labels (str, list[str]) : Names of domain labels to be loaded.
             Default: None.
    """

    def __init__(self,
                 domain_labels=None,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):

        self.with_domain = False
        if domain_labels is not None:
            if isinstance(domain_labels, str):
                self.with_domain = True
                domain_labels = [domain_labels]
            elif isinstance(domain_labels, list):
                assert mmcv.is_list_of(domain_labels, str)
                self.with_domain = True
            else:
                raise ValueError('domain_label must be either str or list of str')
        self.domain_labels = domain_labels

        super(DALoadAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            denorm_bbox=denorm_bbox,
            file_client_args=file_client_args)

    def _load_domain(self, results):
        """Private function to load ground-truth domain adaptive
        annotations for domain adaptation task.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded DA annotations.
        """

        for domain_label in self.domain_labels:
            results[domain_label] = results['ann_info'][domain_label]
            results['domain_fields'].append(domain_label)
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super(DALoadAnnotations, self).__call__(results=results)

        if self.with_domain:
            results = self._load_domain(results)
        return results
