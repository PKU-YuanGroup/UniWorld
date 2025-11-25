import torch
import torch.nn.functional as F
from typing import Literal, Union, Optional, Dict, Tuple, List

class HighFrequencyExtractor:
    _RADIUS_CACHE: Dict[Tuple[torch.device,int,int], torch.Tensor] = {}

    def __init__(
        self,
        downscale_factor: int = 1,
        *,
        anti_alias: bool = True,
        return_abs: bool = False,
        padding_mode: Literal['zeros','reflect','replicate'] = 'reflect',
        low_freq_energy_ratio: Union[float, List[float]] = 0.05,
        center: bool = False,
        standardize: bool = False,
        norm_scope: Literal['per_channel','global'] = 'per_channel',
        normalize_lowpass: bool = False,
        return_lowpass: bool = False,
        allow_grad: bool = False,
        eps: float = 1e-6
    ):
        self.downscale_factor = downscale_factor
        self.anti_alias = anti_alias
        self.return_abs = return_abs
        self.padding_mode = padding_mode
        self.low_freq_energy_ratio = low_freq_energy_ratio

        print(f"Init HighFrequencyExtractor with return_abs {return_abs} and low_freq_energy_ratio {low_freq_energy_ratio}")

        self.center = center
        self.standardize = standardize
        self.norm_scope = norm_scope
        self.normalize_lowpass = normalize_lowpass
        self.return_lowpass = return_lowpass
        self.allow_grad = allow_grad
        self.eps = eps

    def __call__(
        self,
        tensor: torch.Tensor,
        cutoff_freq: Optional[Union[float, torch.Tensor]] = None,
        sigmas: Optional[Union[float, torch.Tensor]] = None
    ):
        return self.extract(tensor, cutoff_freq, sigmas)

    # ----------------- caches -----------------
    @classmethod
    def _get_radius(cls, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (device, h, w)
        r = cls._RADIUS_CACHE.get(key)
        if r is None or r.device != device:
            cy, cx = h // 2, w // 2
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device) - cy,
                torch.arange(w, device=device) - cx,
                indexing='ij'
            )
            r = torch.sqrt(xx.to(torch.float32)**2 + yy.to(torch.float32)**2)
            cls._RADIUS_CACHE[key] = r
        return r

    # ----------------- helpers -----------------
    @staticmethod
    def _complex_mag(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.sqrt(z.real * z.real + z.imag * z.imag + eps)

    @staticmethod
    def _get_low_info_ratio_from_sigma(
        sigmas: torch.Tensor,
        low_threshold: float = 0.1,
        high_threshold: float = 0.8,
        *,
        min_ratio: float = None,
        max_ratio: float = 1.0,
        smooth: bool = True,
        smooth_k: float = 8.0,
    ) -> torch.Tensor:
        if min_ratio is None:
            min_ratio = low_threshold
        if high_threshold <= low_threshold:
            raise ValueError("high_threshold must be greater than low_threshold")
        x = sigmas.to(torch.float32)
        t = (x - low_threshold) / (high_threshold - low_threshold)
        if smooth:
            u = torch.sigmoid(smooth_k * (t - 0.5))
            y = min_ratio + (max_ratio - min_ratio) * u
        else:
            t_lin = t.clamp(0.0, 1.0)
            y = min_ratio + (max_ratio - min_ratio) * t_lin
        return y.clamp(min=min_ratio, max=max_ratio).to(sigmas.dtype)

    def _downsample(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        s = self.downscale_factor
        if s == 1:
            return x
        if self.anti_alias:
            new_h = (h // s) * s
            new_w = (w // s) * s
            if (x.shape[-2] != new_h) or (x.shape[-1] != new_w):
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                                  align_corners=False, antialias=True)
            return F.avg_pool2d(x, kernel_size=s, stride=s)
        else:
            return F.avg_pool2d(x, kernel_size=s, stride=s)

    def _normalize(self, t: torch.Tensor, *, for_lowpass: bool=False) -> torch.Tensor:
        if for_lowpass and not self.normalize_lowpass:
            return t
        if not for_lowpass and not (self.center or self.standardize):
            return t
        b, c = t.shape[:2]
        if self.norm_scope == 'per_channel':
            mean = t.mean(dim=(-2, -1), keepdim=True) if self.center else 0.0
            if self.standardize:
                std = t.flatten(2).std(dim=2, keepdim=True).view(b, c, 1, 1)
                t = (t - mean) / (std + self.eps) if self.center else t / (std + self.eps)
            else:
                t = t - mean
        elif self.norm_scope == 'global':
            mean = t.mean(dim=(1,2,3), keepdim=True) if self.center else 0.0
            if self.standardize:
                std = t.view(b, -1).std(dim=1, keepdim=True).view(b, 1, 1, 1)
                t = (t - mean) / (std + self.eps) if self.center else t / (std + self.eps)
            else:
                t = t - mean
        else:
            raise ValueError("norm_scope must be 'per_channel' or 'global'")
        return t

    # ----------------- core -----------------
    def extract(
        self,
        tensor: torch.Tensor,
        cutoff_freq: Optional[Union[float, torch.Tensor]] = None,
        sigmas: Optional[Union[float, torch.Tensor]] = None
    ):
        ctx = torch.enable_grad() if self.allow_grad else torch.no_grad()
        with ctx:
            if tensor.dim() != 4:
                raise ValueError(f"expect (B,C,H,W)，get {tensor.shape}")
            b, c, h, w = tensor.shape
            if self.downscale_factor < 1:
                raise ValueError("downscale_factor must >= 1")
            dev, dtype = tensor.device, tensor.dtype

            # FFT
            freq_all    = torch.fft.fft2(tensor.to(torch.float32), dim=(-2, -1))     # (B,C,H,W)
            shifted_all = torch.fft.fftshift(freq_all, dim=(-2, -1))
            radius      = self._get_radius(h, w, dev)                                  # (H,W)

            if isinstance(sigmas, torch.Tensor):
                ratio_b = self._get_low_info_ratio_from_sigma(sigmas.to(tensor.device))
            elif isinstance(sigmas, (float, int)):
                s = torch.full((tensor.size(0),), float(sigmas), device=tensor.device)
                ratio_b = self._get_low_info_ratio_from_sigma(s)
            elif isinstance(self.low_freq_energy_ratio, list):
                lo, hi = float(self.low_freq_energy_ratio[0]), float(self.low_freq_energy_ratio[1])
                ratio_b = (torch.rand(b, device=tensor.device) * (hi - lo) + lo)
            else:
                ratio_b = torch.full((b,), float(self.low_freq_energy_ratio), device=tensor.device)
            ratio_b = ratio_b.clamp_(0.0, 1.0)  # (B,)

            if cutoff_freq is None:
                # Calculate the energy per sample: take the mean across channels -> (B,H,W)
                mag2d = self._complex_mag(shifted_all).mean(dim=1)                     # (B,H,W)

                # Flatten and sort the radius (the radius sorting index is consistent across all samples)
                flat_r = radius.flatten()                                              # (N,)
                idx    = torch.argsort(flat_r)                                         # (N,)
                r_sorted = flat_r[idx]                                                 # (N,)

                # Sort the energy of each sample by radius
                mag_flat   = mag2d.view(b, -1)                                         # (B,N)
                mag_sorted = mag_flat[:, idx]                                          # (B,N)

                # Cumulative sum and target energy
                cum = torch.cumsum(mag_sorted, dim=1)                                  # (B,N)
                total = cum[:, -1]                                                     # (B,)
                target = total * ratio_b                                               # (B,)

                # Find the location of the first ≥ target (per sample)
                k = torch.sum(cum < target.unsqueeze(1), dim=1)                        # (B,)
                k = torch.clamp(k, 0, r_sorted.numel() - 1)                           

                # Cut off radius per sample (B,)
                cutoff_vec = r_sorted.index_select(0, k)                               # (B,)
            else:
                # Externally given: can be a scalar or (B,)
                if isinstance(cutoff_freq, torch.Tensor):
                    cutoff_vec = cutoff_freq.to(tensor.device).reshape(-1)
                    if cutoff_vec.numel() == 1:
                        cutoff_vec = cutoff_vec.repeat(b)
                else:
                    cutoff_vec = torch.full((b,), float(cutoff_freq), device=tensor.device)

            # Construct masks for each sample (B, 1, H, W)
            mask_h = (radius.unsqueeze(0) >= cutoff_vec.view(b, 1, 1)).to(torch.float32).unsqueeze(1)  # (B,1,H,W)
            mask_l = 1.0 - mask_h

            Fh = shifted_all * mask_h

            # IFFT -> HF
            hf_c = torch.fft.ifft2(torch.fft.ifftshift(Fh, dim=(-2,-1)), dim=(-2,-1))
            hf   = self._complex_mag(hf_c) if self.return_abs else hf_c.real

            # Low pass
            if self.return_lowpass:
                Fl = shifted_all * mask_l
                lf_c = torch.fft.ifft2(torch.fft.ifftshift(Fl, dim=(-2,-1)), dim=(-2,-1))
                lf = lf_c.real.to(dtype)

            # Normalization + downsampling
            if self.return_lowpass:
                lf = self._normalize(lf, for_lowpass=True)
            hf = self._normalize(hf, for_lowpass=False)

            if self.return_lowpass:
                lf = self._downsample(lf, h, w).to(dtype)
            hf = self._downsample(hf, h, w).to(dtype)

            out = (lf, hf) if self.return_lowpass else hf
            return out if self.allow_grad else (tuple(o.detach() for o in out) if isinstance(out, tuple) else out.detach())
