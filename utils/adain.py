import numpy as np
import torch
from scipy.ndimage import binary_erosion
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma):
    size = content_feat.size()
    normalized_feat = (content_feat - content_mu.expand(size)) / content_sigma.expand(size)
    return normalized_feat * style_sigma.expand(size) + style_mu.expand(size)


def process_mask(feats, binary_mask, nb_class):
    binary_mask = torch.from_numpy(binary_mask).cpu()
    # extract one class from segmentation map
    processed_mask = np.where(binary_mask == nb_class, 1, 0)
    
    processed_mask = binary_erosion(processed_mask)
    # resize mask to match the features
    processed_mask = processed_mask.astype(np.float32)
    
    height = feats.shape[1]
    width = feats.shape[2]
    processed_mask = np.array(Image.fromarray(processed_mask).resize((width, height)))

    processed_mask = torch.from_numpy(processed_mask).cuda()

    return feats, processed_mask


def custom_adain_pixel(content_feat, style_feat, content_label = None, style_label = None):
    torch.manual_seed(42)
    np.random.seed(42)
    
    style_mu = torch.zeros((4, *style_label.shape), dtype=torch.float32).cuda()
    style_sigma = torch.ones((4, *style_label.shape), dtype=torch.float32).cuda()
    
    content_mu = torch.zeros((4, *content_label.shape), dtype=torch.float32).cuda()
    content_sigma = torch.ones((4, *content_label.shape), dtype=torch.float32).cuda()
    
    # 19 classes of cityscapes
    for c in range(19):
        content_feat, content_mask = process_mask(content_feat, content_label, c)
        content_mean, content_std = calc_mean_std_smooth(content_feat, mask=content_mask)
        
        mu, sigma = None, None
        
        mask = content_label == c
        mask = np.repeat(mask[np.newaxis,:,:], 4, axis=0)
        
        # check if content has meaningful statistics about the current class
        if not (torch.isnan(content_mean).any() or torch.isnan(content_std).any()):
            style_feat, style_mask = process_mask(style_feat, style_label, c)
            if style_mask.max() > 0. :
                mu, sigma = calc_mean_std_smooth(style_feat, mask=style_mask)
                
            if mu is not None and sigma is not None:
                style_mu = style_mu + mu * content_mask
                style_sigma = style_sigma + sigma * content_mask
                
                content_mu = content_mu + content_mean * content_mask
                content_sigma = content_sigma + content_std * content_mask
    
    content_feat = adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma)
    
    # content feat is squeezed during the preprocessing 
    content_feat = content_feat.unsqueeze(0)
    return content_feat

def calc_mean_std_smooth(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    
    mask = mask.view(-1)  # Flatten the mask
    feat_flat = feat.view(C, -1)  # Flatten the feature tensor

    # Compute weighted mean
    weighted_sum = (feat_flat * mask).sum(dim=1)
    sum_of_weights = mask.sum()
    weighted_mean = (weighted_sum / sum_of_weights).view(C, 1, 1)

    # Compute weighted variance
    squared_diff = (feat_flat - weighted_mean.view(C, -1)) ** 2
    weighted_variance = (squared_diff * mask).sum(dim=1) / sum_of_weights
    weighted_variance += eps  # Add epsilon for numerical stability
    weighted_std = weighted_variance.sqrt().view(C, 1, 1)

    return weighted_mean, weighted_std

def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)
    
    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std


import torch.nn.functional as F


def _weighted_stats(feat, w, eps=1e-5):
    """Same math as calc_mean_std_smooth, on a soft weight map. feat: [C,H,W], w: [1,H,W]."""
    n = w.sum().clamp(min=1e-8)
    mu = (feat * w).sum(dim=(-2, -1), keepdim=True) / n
    var = (w * (feat - mu) ** 2).sum(dim=(-2, -1), keepdim=True) / n
    return mu, (var + eps).sqrt()


def _erode(w, k=3):
    """Grayscale erosion (min-pool); counterpart of binary_erosion in process_mask."""
    return -F.max_pool2d(-w[None], k, stride=1, padding=k // 2)[0]


def _class_stats(feat, w, min_weight, erode):
    """Class stats as in custom_adain_pixel: eroded mask if enough weight survives, and
    None (-> identity) if the class has too little support to give meaningful statistics."""
    if erode:
        w_er = _erode(w)
        if w_er.sum() >= min_weight:
            w = w_er
    if w.sum() < min_weight:
        return None
    return _weighted_stats(feat, w)


def runway_adain(content_feat, style_feat, content_mask, style_mask=None,
                 strength=1.0, min_weight=8.0, erode=True):
    """Class-wise AdaIN (Eqs. 3-5) with classes {runway, background}.
    content_feat, style_feat: [C,H,W]; masks: soft [1,H,W] in [0,1] at latent resolution.
    strength: AdaIN strength inside the runway (1 = full Eq. 5, 0 = none).
    Classes lacking statistics are left unchanged, as in custom_adain_pixel; the exception is
    the background of a style image with no annotation, which uses global stats (Eq. 2)."""
    x, s = content_feat.float(), style_feat.float()
    w_r = content_mask.to(x.device, x.dtype).clamp(0, 1)
    w_b = 1.0 - w_r
    ones = torch.ones_like(w_r)

    def class_out(w_c, w_s):
        cs = _class_stats(x, w_c, min_weight, erode)
        ss = _class_stats(s, w_s, min_weight, erode) if w_s is not None else None
        if cs is None or ss is None:
            return x                                                   # identity
        (mu_x, sd_x), (mu_y, sd_y) = cs, ss
        return sd_y * (x - mu_x) / sd_x + mu_y                         # Eq. 5

    m_s = None if style_mask is None else style_mask.to(x.device, x.dtype).clamp(0, 1)
    out_r = class_out(w_r, m_s)                                        # no style runway -> identity
    out_r = x + strength * (out_r - x)
    out_b = class_out(w_b, ones if m_s is None else 1.0 - m_s)         # no style mask -> global

    return (w_r * out_r + w_b * out_b).to(content_feat.dtype)          # soft blend, no seams