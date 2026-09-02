from pathlib import Path

import torch


def _load_midas():
    """Load MiDaS from Torch Hub's local cache without a GitHub probe."""
    hub_dir = Path(torch.hub.get_dir())
    for repo_name in ("intel-isl_MiDaS_master", "intel-isl_MiDaS_main"):
        repo_dir = hub_dir / repo_name
        if (repo_dir / "hubconf.py").is_file():
            print(f"[MiDaS] Loading local Torch Hub cache: {repo_dir}")
            return torch.hub.load(str(repo_dir), "DPT_Hybrid", source="local")

    # Keep the first-run download behavior, but pin the branch so Torch Hub does
    # not issue a separate request just to discover whether it is main/master.
    return torch.hub.load(
        "intel-isl/MiDaS:master",
        "DPT_Hybrid",
        trust_repo=True,
        skip_validation=True,
    )


midas = _load_midas()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

downsampling = 1


def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

