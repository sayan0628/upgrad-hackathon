"""Export the cross-encoder reranker to ONNX (fp32) and dynamic INT8, and save its tokenizer, under models/reranker/.

    python tools/export_reranker_onnx.py

Outputs (large binaries, not committed anywhere else):
    models/reranker/model_fp32.onnx   dynamic batch and sequence axes, raw logits out (sigmoid is applied at runtime)
    models/reranker/model_int8.onnx   onnxruntime dynamic INT8 quantization of the fp32 graph (MatMul weights)
    models/reranker/tokenizer/        HF tokenizer files, so the ONNX reranker needs no torch at serving time
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kestrel.retrieval.rerank import CROSS_ENCODER_DEFAULT  # noqa: E402

OUT = ROOT / "models" / "reranker"
OPSET = 17   # onnxruntime 1.32 supports opset 17-23; 17 is what the legacy TorchScript exporter emits cleanly


def main() -> None:
    from sentence_transformers import CrossEncoder
    from onnxruntime.quantization import QuantType, quantize_dynamic

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    ce = CrossEncoder(CROSS_ENCODER_DEFAULT, device="cpu", max_length=512)
    model = ce.model.eval()
    tok = ce.tokenizer
    tok.save_pretrained(OUT / "tokenizer")

    enc = tok(["what is the deductible?", "a longer second text"], ["passage one about deductibles.", "passage two"],
              padding=True, truncation=True, max_length=64, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc]
    args = tuple(enc[n] for n in names)

    fp32 = OUT / "model_fp32.onnx"
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, args, str(fp32), input_names=names, output_names=["logits"], opset_version=OPSET,
            dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "logits": {0: "batch"}},
            do_constant_folding=True, dynamo=False)
    print("exported", fp32, round(fp32.stat().st_size / 1e6, 1), "MB")

    int8 = OUT / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print("quantized", int8, round(int8.stat().st_size / 1e6, 1), "MB")

    import onnxruntime as ort
    for p in (fp32, int8):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = s.run(None, {n: enc[n].numpy() for n in names})[0]
        ref = model(**enc).logits.detach().numpy()
        print(p.name, "sanity max|onnx-torch| logits:", float(np.max(np.abs(out - ref))))


if __name__ == "__main__":
    main()
