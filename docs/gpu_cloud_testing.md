# GPU Cloud Testing

PLM-14 adds a no-file-shuffling GPU check for pcketlm. The loop is:

1. Push this repo to GitHub.
2. Open `notebooks/gpu_test.ipynb` in Google Colab or Kaggle.
3. Set the runtime to a free GPU.
4. Run all cells.
5. Paste the final JSON and pytest line back into the Linear ticket.

## GitHub Setup

This checkout currently has no git remote configured. The operator needs to create a private GitHub repo once, then run:

```powershell
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin plm-14-gpu-testing-pipeline
```

For a private repo in Colab, create a GitHub token with read access to the repo. The notebook asks for it with `getpass`, so it is not stored in the notebook.

## Notebook Settings

In Colab:

1. Open `notebooks/gpu_test.ipynb`.
2. Runtime -> Change runtime type -> T4 GPU or another free GPU.
3. Set:

```python
REPO_URL = "https://github.com/<owner>/<repo>.git"
BRANCH = "plm-14-gpu-testing-pipeline"
```

Then click Run all.

## What The Smoke Test Covers

The cloud smoke is intentionally small and synthetic. It does not download DeepSeek V3. It checks that a GPU runtime can execute the FP8-shaped pieces we need before bigger CUDA work begins:

- FP8 e4m3 byte tensors with per-block fp32 scales.
- FP8 dequantization into torch float tensors.
- MLP gate/up/down math.
- MoE top-k routing and expert combine.
- MLA-shaped attention over latent and RoPE caches.

The local CPU path remains unchanged. On a CPU-only machine, `tests/gpu/test_gpu_smoke_cuda.py` skips cleanly. The normal CPU fallback smoke still runs through:

```powershell
python tools/gpu_smoke.py --json
```

For a real cloud GPU gate, use:

```bash
python tools/gpu_smoke.py --require-cuda --json
python -m pytest tests/gpu -q
```
