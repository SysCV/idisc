from typing import List

import torch

DICT_METRICS_DEPTH = {
    "d05": "max",
    "d1": "max",
    "d2": "max",
    "d3": "max",
    "rmse": "min",
    "rmse_log": "min",
    "abs_rel": "min",
    "sq_rel": "min",
    "log10": "min",
    "silog": "min",
}

DICT_METRICS_NORMALS = {
    "a1": "max",
    "a2": "max",
    "a3": "max",
    "a4": "max",
    "a5": "max",
    "rmse_angular": "min",
    "mean": "min",
    "median": "min",
}


class RunningMetric(object):
    def __init__(self, metrics_names: List[str]):
        super().__init__()
        self.num_samples = 0.0
        self.metrics_dict = {name: 0.0 for name in metrics_names}

    def accumulate_metrics(self, gt, pred, mask=None):
        # get splits as batch elements if not mask, if mask we need indeces
        new_samples = splits = gt.shape[0]
        splits = tuple((c_gt.flatten().size(0) for c_gt in gt))
        if mask is not None:
            mask = mask.bool()
            pred = pred[mask]
            gt = gt[mask]
            splits = tuple(mask.reshape(mask.shape[0], -1).sum(dim=-1).cpu().numpy())

        for k, v in self.metrics_dict.items():
            self.metrics_dict[k] = globals()[k](
                gt, pred, v, self.num_samples, new_samples, splits
            )
        self.num_samples += new_samples

    def get_metrics(self):
        try:
            return {k: v.detach().cpu().item() for k, v in self.metrics_dict.items()}
        except ValueError:
            return self.metrics_dict

    def reset_metrics(self):
        for k, v in self.metrics_dict.items():
            self.metrics_dict[k] = 0.0
        self.num_samples = 0.0


def angular_err(gt, pred):
    prediction_error = torch.cosine_similarity(gt, pred, dim=1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    err = torch.acos(prediction_error) * 180.0 / torch.pi
    return err


def cumulate_mean(new_val, stored_val, new_samples, stored_samples):
    new_ratio = new_samples / (stored_samples + new_samples)
    storage_ratio = stored_samples / (stored_samples + new_samples)
    return new_val * new_ratio + stored_val * storage_ratio


def d05(gt, pred, stored_value, stored_samples, new_samples, splits):
    thresh = torch.maximum((gt / pred), (pred / gt))
    update_value = cumulate_mean(
        (thresh < 1.25**0.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def d1(gt, pred, stored_value, stored_samples, new_samples, splits):
    thresh = torch.maximum((gt / pred), (pred / gt))
    update_value = cumulate_mean(
        (thresh < 1.25).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def d2(gt, pred, stored_value, stored_samples, new_samples, splits):
    thresh = torch.maximum((gt / pred), (pred / gt))
    update_value = cumulate_mean(
        (thresh < 1.25**2).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def d3(gt, pred, stored_value, stored_samples, new_samples, splits):
    thresh = torch.maximum((gt / pred), (pred / gt))
    update_value = cumulate_mean(
        (thresh < 1.25**3).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def rmse(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        torch.sqrt(((gt - pred) ** 2 + 1e-6).mean())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def rmse_log(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2 + 1e-6).mean())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def abs_rel(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        (torch.abs(gt - pred) / gt).mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def sq_rel(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        (((gt - pred) ** 2) / gt).mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def log10(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def silog(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        100 * torch.sqrt((torch.log(pred) - torch.log(gt)).var())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def rmse_angular(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        torch.sqrt((angular_err(gt, pred) ** 2 + 1e-6).mean())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def mean(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(err.mean(), stored_value, new_samples, stored_samples)
    return update_value


def median(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        err.median(), stored_value, new_samples, stored_samples
    )
    return update_value


def a1(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a2(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 7.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a3(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 11.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a4(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 22.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a5(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 30.0).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value
