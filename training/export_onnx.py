"""Export a fine-tuned CrossEncoder to ONNX (optionally int8-quantized).

Quantized ONNX keeps the fine-tuned Indonesian reranker small and CPU-fast so it
stays viable on a cheap VPS. The output directory contains ``model.onnx`` plus
the tokenizer files, ready to be served by flashindorank's CustomReranker.

Run:
    python training/export_onnx.py --model-dir models/ft-id-ce --out-dir models/ft-id-ce-onnx
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/ft-id-ce")
    parser.add_argument("--out-dir", default="models/ft-id-ce-onnx")
    parser.add_argument("--no-quantize", action="store_true", help="skip int8 quantization")
    args = parser.parse_args()

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The CrossEncoder save dir nests the HF model; fall back to the dir itself.
    src = args.model_dir
    print(f"Exporting {src} -> ONNX ...")
    model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    tokenizer = AutoTokenizer.from_pretrained(src)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Normalize the ONNX filename to model.onnx for the serving layer.
    onnx_files = list(out_dir.glob("*.onnx"))
    if onnx_files and onnx_files[0].name != "model.onnx":
        shutil.copy(onnx_files[0], out_dir / "model.onnx")

    if not args.no_quantize:
        try:
            from optimum.onnxruntime import AutoQuantizationConfig, ORTQuantizer

            print("Applying dynamic int8 quantization ...")
            quantizer = ORTQuantizer.from_pretrained(out_dir, file_name="model.onnx")
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
            quant = list(out_dir.glob("*quantized*.onnx")) or list(out_dir.glob("*_quant*.onnx"))
            if quant:
                shutil.copy(quant[0], out_dir / "model.onnx")
                print(f"Quantized model -> {out_dir/'model.onnx'}")
        except Exception as e:  # noqa: BLE001
            print(f"Quantization skipped ({type(e).__name__}: {e}); keeping fp32 model.onnx")

    size_mb = (out_dir / "model.onnx").stat().st_size / 1e6
    print(f"Done. Serving model: {out_dir/'model.onnx'} ({size_mb:.1f} MB)")
    print("Tokenizer files:", [p.name for p in out_dir.glob('*token*')] + [p.name for p in out_dir.glob('*spiece*')])


if __name__ == "__main__":
    main()
