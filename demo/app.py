"""Gradio demo: upload a batik photo, get the top-3 Semarang motifs.

Run:  pip install gradio && python demo/app.py [--checkpoint PATH]
The model is a closed-set classifier over 10 motifs; it will always answer with
one of them, so out-of-scope images must be judged by a human (see README).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batik.predict import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/release/batik_effb0_final.pt")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    import gradio as gr

    ckpt = Path(a.checkpoint)
    ckpt = ckpt if ckpt.is_absolute() else Path(__file__).resolve().parents[1] / ckpt
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found: run `python -m batik release` after training (or download the release asset)")
    predictor = Predictor(ckpt)

    def classify(img):
        if img is None:
            return {}
        return {c: p for c, p in predictor(img, k=len(predictor.classes))}

    gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Foto kain batik"),
        outputs=gr.Label(num_top_classes=3, label="Prediksi motif"),
        title="Klasifikasi Motif Batik Semarang (EfficientNet-B0, fine-tuned)",
        description="Closed-set 10 motif. Prediksi wajib diverifikasi kurator; confidence rendah = periksa manual.",
    ).launch(share=a.share)


if __name__ == "__main__":
    main()
