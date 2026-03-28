# Flash Attention 3 for Blackwell (sm_100)

## Background

By default the training job uses `attention_mechanism: cudnn` in `k8s/job/configmap.yaml`,
which gives approximately 20s/step on the GB10. Flash Attention 3 is specifically optimized
for Hopper (sm_90) and Blackwell (sm_100) and can realistically halve that to ~10-13s/step,
cutting a 3000-step run from ~17 hours to ~9 hours.

FA3 is not included in the SimpleTuner pip install or the NGC base image. It must be compiled
from source inside the container image.

## Caveats

**The build is officially unsupported.** The SimpleTuner install docs state:

> "Managing the flash_attn build is poorly-supported in SimpleTuner, currently. This can
> break on updates, requiring you to re-run this build procedure manually from time to time."

Specific risks:

1. **Compilation against NGC PyTorch.** FA3 compiles C++/CUDA extensions against the
   installed PyTorch and CUDA toolkit. The NGC image ships its own PyTorch build
   (`pytorch_triton`, custom ops) that may not match what flash-attention expects. The build
   may fail or silently produce a broken binary.

2. **SimpleTuner updates can break it.** If SimpleTuner upgrades its diffusers or PyTorch
   dependency in a way that changes the attention dispatch interface, the FA3 build becomes
   stale and needs to be rebuilt. There is no automatic detection of this — training will
   either error on startup or silently fall back.

3. **Build time.** Compiling FA3 CUDA kernels for sm_100 takes 10-30 minutes during
   `docker build`. Factor this into your build pipeline.

4. **ARM64 / aarch64.** Flash Attention 3 has limited testing on ARM64 SBSA (Grace CPU).
   The build may succeed but produce slower kernels than on x86_64, or fail entirely.
   cuDNN (`attention_mechanism: cudnn`) is the safer fallback if FA3 underperforms.

5. **Driver compatibility.** FA3 with sm_100 requires CUDA 13.0+. The current base image
   (`nvcr.io/nvidia/pytorch:26.03-py3`) satisfies this, but upgrading beyond 26.x may
   require driver >= 590. See AGENTS.md for driver constraints.

## How to Build

### 1. Add the FA3 build step to the Dockerfile

Add the following block to `Dockerfile` after the SimpleTuner install step:

```dockerfile
# Build Flash Attention 3 for Blackwell (sm_100).
# See docs/flash-attention-3.md for caveats before enabling this.
RUN git clone --depth=1 https://github.com/Dao-AILab/flash-attention /tmp/flash-attention && \
    cd /tmp/flash-attention/hopper && \
    python setup.py install && \
    rm -rf /tmp/flash-attention
```

### 2. Switch the attention backend in the configmap

In `k8s/job/configmap.yaml`, change:

```json
"attention_mechanism": "cudnn",
```

to:

```json
"attention_mechanism": "flash-attn-3",
```

Apply before the next job run:

```bash
kubectl apply -f k8s/job/configmap.yaml -n openclaw
```

No image rebuild is needed for the configmap change alone — but the image must have been
built with the FA3 step above for `flash-attn-3` to work.

### 3. Verify it's being used

After the job starts, look for this in the logs:

```
[INFO] Patched Attention with flexible fusion (permanent=True)
```

And confirm no fallback warning like:

```
[WARNING] Attention backend 'flash-attn-3' is unavailable
```

If FA3 is unavailable, SimpleTuner will error on startup rather than silently fall back,
so a clean startup is confirmation it's active.

## Reverting

If the FA3 build causes issues, revert both changes:

1. Remove the `git clone` block from `Dockerfile` and rebuild the image.
2. Change `attention_mechanism` back to `"cudnn"` in `configmap.yaml` and `kubectl apply`.

cuDNN at ~20s/step is a reliable baseline that requires no custom build.
