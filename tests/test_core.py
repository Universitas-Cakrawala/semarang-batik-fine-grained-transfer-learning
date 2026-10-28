import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from batik.cli import matrix_order
from batik.config import load_config
from batik.data.audit import canonical_class
from batik.data.audit import dihedral_views
from batik.data.split import _d8, _hamming_matrix, assign_splits, build_groups, dihedral_distance
from batik.data.transforms import eval_transform, train_transform
from batik.metrics import compute_metrics, paired_bootstrap, topk_accuracy
from batik.models import CustomCNN, build_model, count_parameters, freeze_backbone, unfreeze_from
from batik.train import run_id_for, stage_plan


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/experiments/G3_effnetb0_pretrained_aug.yaml")


def test_config_inheritance(cfg):
    assert cfg["group"] == "G3" and cfg["model"]["pretrained"] is True
    assert cfg["train"]["lr"] == 3e-4 and cfg["train"]["epochs"] == 30  # inherited from base
    g4 = load_config("configs/experiments/G4_effnetb0_pretrained_noaug.yaml")
    # G3 vs G4 must differ ONLY in augmentation (single-factor ablation)
    strip = lambda c: {k: v for k, v in c.items() if k not in ("group", "description", "_source", "augmentation")}
    assert strip(cfg) == strip(g4)
    assert cfg["augmentation"]["enabled"] and not g4["augmentation"]["enabled"]


def test_m2_m3_differ_only_in_init():
    g2 = load_config("configs/experiments/G2_effnetb0_scratch_aug.yaml")
    g3 = load_config("configs/experiments/G3_effnetb0_pretrained_aug.yaml")
    assert g2["train"] == g3["train"] and g2["augmentation"] == g3["augmentation"]
    assert g2["model"] | {"pretrained": True} == g3["model"]


def test_custom_cnn_matches_plan():
    m = CustomCNN()
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    assert [c.out_channels for c in convs] == [32, 64, 128, 256]
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 10)
    assert count_parameters(m)["total"] < 500_000


def test_two_stage_freezing():
    cfg = load_config("configs/experiments/G5_effnetb0_pretrained_twostage.yaml")
    cfg["model"]["pretrained"] = False  # no download needed for the test
    m = build_model(cfg)
    freeze_backbone(m)
    assert count_parameters(m)["trainable"] == 1280 * 10 + 10
    unfreeze_from(m, 5)
    assert all(not p.requires_grad for p in m.features[4].parameters())
    assert all(p.requires_grad for p in m.features[8].parameters())
    plan = stage_plan(cfg)
    assert sum(s["epochs"] for s in plan) == cfg["train"]["epochs"]


def test_transforms_shapes_and_determinism(cfg):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (320, 427, 3), dtype=np.uint8))
    ev = eval_transform(cfg)
    assert ev(img).shape == (3, 224, 224) and torch.equal(ev(img), ev(img))
    tr = train_transform(cfg)
    assert tr(img).shape == (3, 224, 224) and not torch.equal(tr(img), tr(img))
    off = dict(cfg, augmentation={**cfg["augmentation"], "enabled": False})
    assert torch.equal(train_transform(off)(img), ev(img))


def test_metrics():
    probs = np.eye(10)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 8]]
    labels = np.arange(10)
    m = compute_metrics(probs, labels, [str(i) for i in range(10)])
    assert m["accuracy"] == pytest.approx(0.9)
    assert topk_accuracy(probs, labels, 1) == pytest.approx(0.9)
    assert np.array(m["confusion_matrix"]).sum() == 10
    b = paired_bootstrap(labels, labels, probs.argmax(1), 10, n_boot=200)
    assert b["observed_diff"] > 0


def test_hamming_and_groups():
    h = ["ffffffffffffffff", "fffffffffffffffe", "0000000000000000"]
    ham = _hamming_matrix(h)
    assert ham[0, 1] == 1 and ham[0, 2] == 64
    frame = pd.DataFrame({"class": ["a", "a", "a"], "phash": h, "relpath": ["x", "y", "z"]})
    emb = np.eye(3, dtype=np.float32)
    groups, edges = build_groups(frame, emb, phash_max=8, cos_min=0.97)
    assert groups[0] == groups[1] != groups[2] and len(edges) == 1


def test_group_split_keeps_groups_together():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 300)
    groups = np.arange(3000) // 3  # triplets of near-duplicates
    split = assign_splits(labels, groups, seed=42, n_folds=20, fractions={"train": .7, "val": .15, "test": .15})
    df = pd.DataFrame({"g": groups, "s": split, "y": labels})
    assert df.groupby("g").s.nunique().max() == 1
    frac = df.s.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02 and abs(frac["test"] - 0.15) < 0.02
    again = assign_splits(labels, groups, seed=42, n_folds=20, fractions={"train": .7, "val": .15, "test": .15})
    assert (again == split).all()


def test_matrix_priority_and_run_ids(cfg):
    order = matrix_order(["G1", "G2", "G3", "G4", "G1g", "G2g", "G3g", "G5"], [42, 43, 44])
    assert order[:9] == [(g, s) for s in (42, 43, 44) for g in ("G1", "G2", "G3")]
    assert order[9] == ("G4", 42) and order[12] == ("G1g", 42) and order[-1] == ("G5", 44) and len(order) == 24
    assert run_id_for(cfg, 42) == "G3_effb0-pt_augon_s42"


def test_class_names():
    assert canonical_class("asem_arang") == "Asem Arang"
    assert canonical_class("Warak  Beras Utah") == "Warak Beras Utah"
    assert canonical_class("Batik Cap Asem Arang") == "Asem Arang"
    assert canonical_class("random") is None


def test_rotated_copies_and_same_capture_are_grouped():
    import imagehash

    rng = np.random.default_rng(1)
    a = Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).resize((256, 192))
    b = Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).resize((256, 192))
    c = Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)).resize((256, 192))
    imgs = [a, a.rotate(90, expand=True), b, c]
    d8 = ["|".join(str(imagehash.phash(v)) for v in dihedral_views(im)) for im in imgs]
    dist = dihedral_distance(_d8(d8), _d8(d8))
    assert dist[0, 1] <= 2 and dist[0, 2] > 8  # rotation found, unrelated image not
    frame = pd.DataFrame({"class": ["k"] * 4, "relpath": list("abcd"), "phash_d8": d8,
                          "datetime_original": ["t1", "t2", "t3", "t3"]})
    groups, _ = build_groups(frame, np.eye(4, dtype=np.float32), phash_max=8, cos_min=0.95)
    assert groups[0] == groups[1]          # rotated copy
    assert groups[2] == groups[3]          # same capture second
    assert groups[0] != groups[2]


def test_gray_protocol_differs_only_in_preprocessing():
    for a, b in [("G1_custom_cnn_scratch_aug", "G1g_custom_cnn_scratch_aug_gray"),
                 ("G3_effnetb0_pretrained_aug", "G3g_effnetb0_pretrained_aug_gray")]:
        ca, cb = (load_config(f"configs/experiments/{x}.yaml") for x in (a, b))
        strip = lambda c: {k: v for k, v in c.items() if k not in ("group", "description", "_source", "preprocess")}
        assert strip(ca) == strip(cb)
        assert cb["preprocess"] == {**ca["preprocess"], "grayscale_equalize": True}
    cg = load_config("configs/experiments/G3g_effnetb0_pretrained_aug_gray.yaml")
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (320, 400, 3), dtype=np.uint8))
    x = eval_transform(cg)(img)
    assert x.shape == (3, 224, 224)
    raw = x * torch.tensor(cg["preprocess"]["std"]).view(3, 1, 1) + torch.tensor(cg["preprocess"]["mean"]).view(3, 1, 1)
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[1], raw[2], atol=1e-5)


def test_protocol_commands_refuse_budget_overrides():
    from batik.cli import main

    for argv in (["matrix", "--epochs", "2"], ["reproduce", "--batch-size", "8"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert "not allowed" in str(e.value)


def test_aggregation_ignores_tagged_runs(tmp_path):
    from batik.analysis import load_runs
    from batik.utils import write_json

    for rid, group in [("G3_effb0-pt_augon_s42", "G3"), ("smoke_G3_effb0-pt_augon_s42", "G3"),
                       ("gpu_G1_customcnn-scratch_augon_s42", "G1"), ("G1g_customcnn-scratch_augon_s42", "G1g")]:
        write_json({"run_id": rid, "group": group, "seed": 42, "subset": None}, tmp_path / rid / "summary.json")
    assert sorted(load_runs(tmp_path).run_id) == ["G1g_customcnn-scratch_augon_s42", "G3_effb0-pt_augon_s42"]


def test_resume_state_loads_on_cpu(tmp_path):
    """RNG states saved in last.pt must be restorable whatever the training device."""
    state = {"rng_cpu": torch.get_rng_state(), "rng_numpy": np.random.get_state()}
    torch.save(state, tmp_path / "last.pt")
    loaded = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(loaded["rng_cpu"])  # raises if the tensor is not a CPU ByteTensor
    src = open("src/batik/train.py").read()
    assert 'torch.load(last_ckpt, map_location="cpu"' in src
