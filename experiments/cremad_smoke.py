"""CREMA-D pipeline smoke test (Phase 10). Tiny end-to-end validation on REAL media:
audio load, video load, forward, backward, feature-norm (rho) logging, gradient
logging, scale-injection (R3 mechanism), checkpoint save/load, and an actor-leakage
assertion. Does NOT run R1/R2/R3. Reuses the frozen analysis helpers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn as nn

from analysis.metrics import contribution_metrics
from data.crema_d import CremadDataset
from interventions.diagnostic import measure_rho
from models.cremad_model import CremadMultimodalModel, count_params
from msc.device import device_report, get_device
from msc.seed import set_seed


def collate(batch):
    a = torch.stack([b[0] for b in batch])       # [B,1,64,T]
    v = torch.stack([b[1] for b in batch])       # [B,8,1,64,64]
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    actors = [b[3] for b in batch]
    return a, v, y, actors


def load_split(ds, device):
    a, v, y = [], [], []
    actors = []
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=0)
    for ba, bv, by, bac in dl:
        a.append(ba); v.append(bv); y.append(by); actors += list(bac)
    return (torch.cat(a).to(device), torch.cat(v).to(device), torch.cat(y).to(device)), actors


def main():
    set_seed(0)
    device = get_device({"type": "cuda", "allow_cpu": False})
    print("Device:", device_report(device))
    report = {}

    man = pd.read_csv(ROOT / "data" / "cremad" / "cremad_manifest.csv")
    present = man[(man.audio_status == "present") & (man.video_status == "present")]
    report["media_present_clips"] = int(len(present))
    if len(present) < 30:
        print("SMOKE ABORT: not enough present media", len(present)); sys.exit(2)

    t0 = time.time()
    tr_ds = CremadDataset(present, "train", limit=24)
    va_ds = CremadDataset(present, "val", limit=12)
    (a_tr, v_tr, y_tr), tr_actors = load_split(tr_ds, device)
    (a_va, v_va, y_va), va_actors = load_split(va_ds, device)
    load_t = time.time() - t0
    print(f"loaded train {a_tr.shape[0]} / val {a_va.shape[0]}  in {load_t:.1f}s")
    print(f"audio batch {tuple(a_tr.shape)}  video batch {tuple(v_tr.shape)}")

    # leakage: train/val actor-disjoint
    leak = set(tr_actors) & set(va_actors)
    report["actor_leakage"] = len(leak)
    assert not leak, f"ACTOR LEAKAGE in smoke subset: {leak}"

    n_classes = 6
    model = CremadMultimodalModel(n_classes, rep_dim=64, fusion="sum", normalization="none").to(device)
    report["params"] = count_params(model)
    report["params_audio"] = sum(p.numel() for p in model.enc_a.parameters())
    report["params_video"] = sum(p.numel() for p in model.enc_v.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()

    # a few training steps with feature-norm + gradient logging
    losses = []
    for step in range(5):
        model.train(); opt.zero_grad()
        logits, (h_a, h_v) = model(a_tr, v_tr, return_reps=True)
        loss = loss_fn(logits, y_tr)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        ga = sum(float(p.grad.pow(2).sum()) for p in model.enc_a.parameters()) ** 0.5
        gv = sum(float(p.grad.pow(2).sum()) for p in model.enc_v.parameters()) ** 0.5
        opt.step()
        losses.append(float(loss.detach()))
    stats = measure_rho(model, a_va, v_va)
    report.update(loss_first=losses[0], loss_last=losses[-1],
                  rho=stats["rho"], mean_norm_audio=stats["mean_norm_a"],
                  mean_norm_video=stats["mean_norm_v"], grad_audio=ga, grad_video=gv,
                  grad_ratio=ga / gv if gv > 0 else float("nan"))

    # modality contribution via masking (frozen helper)
    contrib = contribution_metrics(model, a_va, v_va, y_va, n_classes)
    report["contrib"] = {k: round(v, 3) for k, v in contrib.items()}

    # R3 mechanism check: scale injection on the FROZEN trained reps changes rho
    # predictably. (R3 flips the flag + alpha on the SAME trained model; here we
    # copy weights then set alpha AFTER load so the ScaleInjector buffer isn't reset.)
    m2 = CremadMultimodalModel(n_classes, rep_dim=64, fusion="sum", normalization="none").to(device)
    m2.load_state_dict(model.state_dict(), strict=False)
    m2.scale.enabled = True
    m2.scale.alpha_a.fill_(5.0); m2.scale.alpha_v.fill_(1.0)
    rho_inj = measure_rho(m2, a_va, v_va)["rho"]
    report["rho_after_audio_x5"] = rho_inj
    report["scale_injection_ratio_vs_base"] = rho_inj / stats["rho"] if stats["rho"] > 0 else float("nan")

    # checkpoint save/load
    ckpt = ROOT / "results" / "cremad" / "smoke_ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    m3 = CremadMultimodalModel(n_classes, rep_dim=64).to(device)
    m3.load_state_dict(torch.load(ckpt, map_location=device))
    report["checkpoint_reload_ok"] = True
    ckpt.unlink(missing_ok=True)

    report["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    report["total_runtime_s"] = round(time.time() - t0, 1)

    print("\n===== CREMA-D SMOKE REPORT =====")
    for k, v in report.items():
        print(f"  {k}: {v}")
    ok = (report["actor_leakage"] == 0 and report["checkpoint_reload_ok"]
          and report["loss_last"] <= report["loss_first"] + 1e-3
          and report["scale_injection_ratio_vs_base"] > 3.0)
    print("\nSMOKE:", "PASSED" if ok else "CHECK — see fields above")


if __name__ == "__main__":
    main()
