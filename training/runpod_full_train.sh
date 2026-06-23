#!/usr/bin/env bash
# Full v2 training pipeline for RunPod GPU pods. Logs to /workspace/runpod_train.log
set -euo pipefail
exec > >(tee -a /workspace/runpod_train.log) 2>&1

terminate_pod() {
  if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
    if [[ -f /workspace/DONE ]]; then
      echo "Training succeeded — terminating pod ${RUNPOD_POD_ID}..."
    else
      echo "Training failed — terminating pod ${RUNPOD_POD_ID} to stop billing..."
    fi
    curl -sS -X DELETE \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true
  fi
}
trap terminate_pod EXIT

echo "=== flashIndorank RunPod full train $(date -Iseconds) ==="
cd /workspace

export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "ERROR: HF_TOKEN not set"
  exit 1
fi

if [[ ! -f training/train.py ]]; then
  BRANCH="${FLASHINDORANK_BRANCH:-cursor/id-reranker-v2-6887}"
  rm -rf /workspace/repo
  git clone --depth 1 -b "$BRANCH" https://github.com/madebyaris/flashIndorank.git /workspace/repo
  shopt -s dotglob && cp -a /workspace/repo/* /workspace/ && shopt -u dotglob
fi

# Image ships CUDA torch — pin it so pip cannot replace with CPU build.
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "torch ${TORCH_VER} cuda=$(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "torch==${TORCH_VER}" > /tmp/constraints.txt
pip install -q -r requirements-runpod-gpu.txt -c /tmp/constraints.txt

python3 -c "from transformers import PreTrainedModel; from sentence_transformers import CrossEncoder; print('imports OK')"

echo "--- Step 1: prepare_data (TyDi + MIRACL) ---"
python training/prepare_data.py --sources tydi,miracl --train-negatives 15

echo "--- Step 2: train (full data, 1 epoch) ---"
python training/train.py --epochs 1 --batch-size 32 --out-dir models/ft-id-ce-v2

echo "--- Step 3: TyDi eval ---"
python training/evaluate.py --models \
  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  models/ft-id-ce-v2 | tee /workspace/eval_tydi.txt

echo "--- Step 4: MIRACL eval ---"
python training/eval_miracl.py --model models/ft-id-ce-v2 | tee /workspace/eval_miracl.txt

echo "--- Step 5: export ONNX ---"
python training/export_onnx.py --model-dir models/ft-id-ce-v2 --out-dir models/ft-id-ce-v2-onnx

echo "--- Step 6: upload to Hugging Face (huggingface branch) ---"
python training/upload_to_hf.py --revision huggingface

touch /workspace/DONE
echo "=== ALL DONE $(date -Iseconds) ==="
