import numpy as np

from bowtrack.annotations import MODEL_FILE, AnnotationStore
from bowtrack.dataset import split_entries


def _entries(counts):
    return [{"file": f"frames/take_{t}_{i:06d}.jpg"} for t, n in counts.items() for i in range(n)]


def test_split_few_takes_keeps_every_take_in_training():
    tr, va = split_entries(_entries({"a": 33, "b": 11, "c": 6}))
    take = lambda e: e["file"].rsplit("_", 1)[0]
    assert {take(e) for e in tr} == {take(e) for e in va} == {f"frames/take_{t}" for t in "abc"}
    assert len(tr) + len(va) == 50 and 5 <= len(va) <= 10
    # held-out frames are the last ones of each take (no neighbour leakage inside a block)
    for t in "abc":
        assert max(e["file"] for e in tr if take(e).endswith(t)) < min(e["file"] for e in va if take(e).endswith(t))


def test_split_many_takes_holds_out_whole_takes():
    tr, va = split_entries(_entries({c: 10 for c in "abcdefgh"}))
    take = lambda e: e["file"].rsplit("_", 1)[0]
    assert not ({take(e) for e in tr} & {take(e) for e in va})


def test_model_labels_are_a_separate_group(tmp_path):
    img = np.zeros((20, 30, 3), np.uint8)
    approved, proposed = AnnotationStore(tmp_path), AnnotationStore(tmp_path, MODEL_FILE)
    proposed.put("frames/x_000001.jpg", img, {"scroll": [1.0, 2.0]}, conf={"scroll": 0.8})
    assert len(AnnotationStore(tmp_path)) == 0 and len(AnnotationStore(tmp_path, MODEL_FILE)) == 1
    assert AnnotationStore(tmp_path, MODEL_FILE).entry("frames/x_000001.jpg")["conf"] == {"scroll": 0.8}
    # approve = copy to the approved store, drop from the model store but keep the image
    approved.put("frames/x_000001.jpg", img, proposed.get("frames/x_000001.jpg"))
    proposed.remove("frames/x_000001.jpg", delete_image=False)
    assert len(AnnotationStore(tmp_path)) == 1 and len(AnnotationStore(tmp_path, MODEL_FILE)) == 0
    assert (tmp_path / "frames/x_000001.jpg").exists()
