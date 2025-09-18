# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

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
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

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
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "save_feats", False) or (getattr(self, "_feats", None) is not None)
        expanded_feats = getattr(self, "expanded_feats", False)
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
            multi_label=True
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats,self._feats2, preds[1], expanded_feats=expanded_feats)
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    """def get_obj_feats(self, feat_maps, idxs, expanded_feats=False):
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch
    """

    def get_obj_feats(self, feat_maps,feat_maps2, idxs, expanded_feats=False):
        """
        feat_maps: list of 3 (or N) tensors [B, C_i, H_i, W_i]
        feat_maps2: tensor containing predicted classes
        idxs:      list of index lists per image
        expanded_feats: if True, output [B, total_anchors, D+C+NC] where
                        D = max_i(C_i) and C = number of scales
                        Made from concat of -class predictions
                                            -input to head
                                            -1-hot encoding of which scales used
        """
        import torch

        #print(feat_maps[0].shape)
        #print(feat_maps2[0].shape)
        #exit()

        if not expanded_feats:
            # original “min‐reduce” path
            s = min(x.shape[1] for x in feat_maps)
            obj = torch.cat([
                x.permute(0,2,3,1)
                .reshape(x.shape[0], -1, s, x.shape[1] // s)
                .mean(dim=-1)
                for x in feat_maps
            ], dim=1)
            return [ feats[i] if len(i) else [] for feats,i in zip(obj, idxs) ]

        # --- expanded path with fully dynamic dims ---
        # 1) figure out the “wide” dimension = the largest channel size among inputs
        target_dim = max(x.shape[1] for x in feat_maps)
        # 2) one-hot size = number of scales
        nc=len(self.model.names)
        all_scales = []
        for scale_idx, x in enumerate(feat_maps):
            B, C, H, W = x.shape
            N = H * W
            # flatten spatial → [B, N, C]
            flat = x.permute(0,2,3,1).reshape(B, N, C)

            # pad up to target_dim
            if C < target_dim:
                pad = flat.new_zeros(B, N, target_dim - C)
                feat_wide = torch.cat([flat, pad], dim=2)
            else:
                feat_wide = flat

            # build a dynamic one-hot of length=num_scales
            one_hot = flat.new_zeros(B, N, 8)
            one_hot[:, :, scale_idx] = 1.0

            # concat → [B, N, target_dim + num_scales]
            all_scales.append(torch.cat([feat_wide, one_hot], dim=2))

        # stack all scales → [B, total_anchors, target_dim + num_scales]
        obj = torch.cat(all_scales, dim=1)
        # pick out only the requested indices per image
        ret=[
            feats[i] if len(i) else feats.new_empty((0, feats.shape[-1]))
            for feats, i in zip(obj, idxs)
        ]
        #print(len(ret))
        #print(ret[0].shape)
        #print(feat_maps2[0].shape)
        #exit()
        B=len(ret)
        ret2=[]
        for b in range(B):
            ret_b = ret[b]  # [N, M(B)]
            idx = idxs[b]  # [M(B)]

            if idx.numel() == 0:
                print("MDB warning empty idx in get_obj_feats")
                # build an empty tensor of shape [0, nc + target_dim + num_scales]
                empty_feat = feat_maps2.new_empty((0, nc + obj.shape[-1]))
                ret2.append(empty_feat)
                continue

            selected_feat = feat_maps2[b][:, idx]  # [110, M(B)]
            selected_feat_T = selected_feat.T
            subtensor = selected_feat_T[:, 4:(4+nc)] # just class scores
            ret2.append(torch.cat([subtensor, ret_b], dim=1))
        return ret2

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

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
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
