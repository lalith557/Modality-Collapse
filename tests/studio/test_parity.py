"""Preprocessing parity: the inference path must reproduce the TRAINING cache bit-for-bit,
and the exportable probe must reproduce the paper's probe."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

IDX = [0, 37, 150, 333, 512, 737]


@pytest.mark.parametrize("i", IDX)
def test_cremad_audio_and_video_match_training_cache(cremad_val, tmp_path, i):
    from data.crema_d import load_video_frames
    from inference.cremad_preprocessor import audio_tensor, load_video_frames_streaming, prepare_audio
    ca = np.load(ROOT / "data/cremad/cache/val_audio.npy", mmap_mode="r")
    cv = np.load(ROOT / "data/cremad/cache/val_video.npy", mmap_mode="r")
    r = cremad_val.iloc[i]
    a = audio_tensor(prepare_audio(Path(r.audio_path), "wav", tmp_path))
    v = load_video_frames_streaming(r.video_path)
    assert a.shape == (1, 64, 188) and v.tensor.shape == (8, 1, 64, 64)
    assert torch.equal(a, torch.from_numpy(np.array(ca[i])).float())
    assert torch.equal(v.tensor, torch.from_numpy(np.array(cv[i])).float())
    assert torch.equal(v.tensor, load_video_frames(r.video_path).half().float())   # streaming == original


def test_lossless_container_equals_wav_path(cremad_val, tmp_path):
    """A FLAC copy of a training WAV must give the identical model input (decoder is not a new pipeline)."""
    from scipy.io import wavfile
    from inference.cremad_preprocessor import audio_tensor, prepare_audio
    from scripts.make_external_samples import encode_audio
    wav = Path(cremad_val.iloc[3].audio_path)
    flac = tmp_path / "x.flac"
    encode_audio(wav, flac, "flac", "flac")
    a = audio_tensor(prepare_audio(wav, "wav", tmp_path))
    b = audio_tensor(prepare_audio(flac, "flac", tmp_path))
    assert torch.equal(a, b)


def _utd_val_trial(k: int):
    keys = np.load(ROOT / "data/utd_mhad/cache/val_keys.npy")
    key = str(keys[k])
    d = glob.glob(str(ROOT.parent / "UTD-MHAD" / "Depth" / "**" / f"{key}_depth.mat"), recursive=True)[0]
    i = glob.glob(str(ROOT.parent / "UTD-MHAD" / "Inertial" / "**" / f"{key}_inertial.mat"), recursive=True)[0]
    return Path(d), Path(i)


@pytest.mark.parametrize("k", [0, 50, 107])
def test_utd_matches_training_cache_for_mat_npy_csv(utd_predictor, tmp_path, k):
    import scipy.io as sio
    from inference.utd_preprocessor import prepare_depth, prepare_inertial
    cd = np.load(ROOT / "data/utd_mhad/cache/val_depth.npy"); ci = np.load(ROOT / "data/utd_mhad/cache/val_inertial.npy")
    d, i = _utd_val_trial(k)
    m, s = utd_predictor.mean, utd_predictor.std
    dep = prepare_depth(d, "mat5", tmp_path); ine = prepare_inertial(i, "mat5", tmp_path, m, s)
    assert torch.equal(dep.tensor, torch.from_numpy(cd[k]).float())
    assert torch.equal(ine.tensor, torch.from_numpy(ci[k]).float())
    np.save(tmp_path / "d.npy", np.moveaxis(sio.loadmat(d)["d_depth"], 2, 0))
    np.savetxt(tmp_path / "i.csv", sio.loadmat(i)["d_iner"], delimiter=",", header="ax,ay,az,gx,gy,gz", comments="")
    assert torch.equal(prepare_depth(tmp_path / "d.npy", "npy", tmp_path).tensor, dep.tensor)
    assert torch.allclose(prepare_inertial(tmp_path / "i.csv", "csv", tmp_path, m, s).tensor, ine.tensor, atol=1e-3)


def test_exportable_probe_matches_paper_probe(cremad_pkg):
    from analysis.probes import linear_probe_accuracy
    from inference.probes import fit_linear_probe, probe_logits
    from msc.seed import set_seed
    g = torch.Generator().manual_seed(1)
    xtr, xte = torch.randn(600, 64, generator=g), torch.randn(200, 64, generator=g)
    ytr, yte = torch.randint(0, 6, (600,), generator=g), torch.randint(0, 6, (200,), generator=g)
    dev = cremad_pkg.device
    set_seed(7, True); ref = linear_probe_accuracy(xtr, ytr, xte, yte, 6, dev)
    set_seed(7, True); p = fit_linear_probe(xtr, ytr, 6, dev)
    acc = float((probe_logits(p, xte.to(dev)).argmax(1).cpu() == yte).float().mean())
    assert acc == pytest.approx(ref, abs=1e-12)
