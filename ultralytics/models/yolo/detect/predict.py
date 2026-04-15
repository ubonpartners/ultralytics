# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def _resolve_head(self):
        """Resolve the PyTorch detection head when available, else return None."""
        backend = self.model
        pt_model = getattr(backend, "model", backend)
        try:
            return pt_model.model[-1]  # DetectionModel/PoseModel -> nn.Sequential head
        except Exception:
            try:
                return pt_model[-1]  # last resort if model itself is subscriptable
            except Exception:
                return None

    def _resolve_attr_nc(self, head=None) -> int:
        """Resolve attribute channel count from head first, then backend metadata."""
        head = head if head is not None else self._resolve_head()
        attr_nc = getattr(head, "attr_nc", 0) if head is not None else 0
        if not attr_nc:
            # TensorRT/ONNX backends may only expose metadata fields.
            attr_nc = getattr(self.model, "nc_attr", 0)
        try:
            return max(int(attr_nc), 0)
        except Exception:
            return 0

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        expanded_feats = getattr(self, "expanded_feats", False)
        pred_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
        if save_feats and expanded_feats:
            # Cache raw head outputs so expanded feats can include class-score context for ReID.
            self._feats_head = pred_tensor
        preds = nms.non_max_suppression(
            pred_tensor,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            attr_nc = self._resolve_attr_nc()

            obj_feats = self.get_obj_feats(
                self._feats,
                getattr(self, "_feats_head", None),
                preds[1],
                expanded_feats,
                len(self.model.names),
                attr_nc,
            )
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, feat_maps2, idxs, expanded_feats=False, nc=0, attr_nc=0):
        """Extract object features from the feature maps.

        Standard path (expanded_feats=False): mean-reduce each scale's feature map to a common
        channel width and concatenate across scales. Used for generic feature storage.

        Expanded path (expanded_feats=True, feat_maps2 provided): builds per-anchor vectors that
        include detection-head outputs (class + attribute logits) alongside padded backbone features
        and a per-scale one-hot code. Used by PoseReID to provide the ReIDAdapter with semantic
        context from the head alongside raw spatial features.
        """
        import torch

        if not expanded_feats or feat_maps2 is None:
            s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
            obj_feats = torch.cat(
                [
                    x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1)
                    for x in feat_maps
                ],
                dim=1,
            )  # mean reduce all vectors to same length
            return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]

        # Expanded path mirrors PoseReID input contract:
        # [class+attr logits | padded backbone feats (512) | 8-dim scale code].
        target_dim = 512
        code_len = 8
        all_scales = []
        for scale_idx, x in enumerate(feat_maps):
            b, c, h, w = x.shape
            n = h * w
            flat = x.permute(0, 2, 3, 1).reshape(b, n, c)
            if c < target_dim:
                pad = flat.new_zeros(b, n, target_dim - c)
                feat_wide = torch.cat([flat, pad], dim=2)
            else:
                feat_wide = flat[:, :, :target_dim]
            one_hot = flat.new_zeros(b, n, code_len)
            one_hot[:, :, min(scale_idx, code_len - 1)] = 1.0
            all_scales.append(torch.cat([feat_wide, one_hot], dim=2))

        obj = torch.cat(all_scales, dim=1)
        ret = [feats[idx] if len(idx) else feats.new_empty((0, feats.shape[-1])) for feats, idx in zip(obj, idxs)]
        ret2 = []
        for bi in range(len(ret)):
            ret_b = ret[bi]
            idx = idxs[bi]
            if idx.numel() == 0:
                empty_feat = feat_maps2.new_empty((0, nc + attr_nc + obj.shape[-1]))
                ret2.append(empty_feat)
                continue
            selected_feat = feat_maps2[bi][:, idx]  # [C, M]
            subtensor = selected_feat.T[:, 4 : (4 + nc + attr_nc)]  # class+attr logits for selected anchors
            ret2.append(torch.cat([subtensor, ret_b], dim=1))
        return ret2

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        attr_nc = self._resolve_attr_nc()
        if attr_nc and pred.shape[1] >= 6 + attr_nc + 1:
            pred = pred[:, :-1]
        attributes = pred[:, 6 : 6 + attr_nc] if attr_nc and pred.shape[1] >= 6 + attr_nc else None
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
        if attributes is not None:
            result.update(attributes=attributes)
        return result
