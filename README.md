# CudaSift (fork) + Python wrapper

A CUDA implementation of the SIFT detector and descriptor, forked from
[Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift) (Mårten Björkman).
The original library extracts SIFT features on the GPU at high speed; see the
upstream repository for the algorithm background and the matching code.

This fork keeps the upstream detector and adds three things:

1. **Score based keypoint selection** instead of detection order truncation.
2. An optional **per octave cap** so every scale is represented.
3. A **pybind11 Python module** that returns keypoints and descriptors as NumPy arrays.

---

## Branches

This fork keeps two branches, so you can use the Python wrapper with or without
the keypoint changes:

- **`score-filter`** (default): the Python wrapper plus score based keypoint
  selection and the optional per octave cap. This is what the rest of this README
  documents.
- **`Pascal`**: the same `cudasift_py` wrapper over the unmodified upstream
  detector, with no score filtering. Its `extract()` has no `use_score_filter` or
  `use_per_octave_cap` flags and keeps the original defaults (`num_octaves=6`,
  `thresh=1.7`). Pull this branch if you want the NumPy API but the original
  CudaSift behavior.

Both branches build the same way and import as `cudasift_py`.

---

## What this fork changes

### 1. Keep the strongest keypoints, not the first ones found

Upstream CudaSift allocates a fixed buffer of `maxPts` keypoints. Detection
appends candidates with `atomicInc`, and once the buffer is full the device
kernel clamps the write index to the last slot:

```cpp
idx = (idx >= maxPts ? maxPts - 1 : idx);   // every overflow point overwrites slot maxPts-1
```

So when an image produces more responses than the buffer holds, the kept set is
whichever keypoints were written first (detection order), and every later
candidate overwrites the same final slot. A strong feature can be discarded just
because it was detected late.

This fork decouples the working capacity from the returned cap:

- `maxWorkPts = 4 * maxPts` candidates are stored on the device without
  overwriting (overflow beyond `maxWorkPts` is dropped, not written onto the last
  slot).
- After detection, each octave is filtered on the host to the top `k` candidates
  by `abs(sharpness)` (the extraction time DoG response) using `std::nth_element`
  followed by `std::sort`, then written back, and the device counters are fixed.

So `extract(...)` returns the **highest response** keypoints up to the cap, not an
arbitrary detection ordered subset. The original behavior is still available (see
modes below).

### 2. Optional per octave cap

With the per octave cap enabled, the `maxPts` budget is split evenly across
octaves (the remainder goes to the finest octaves). Each scale gets a guaranteed
quota, so features are selected across the whole scale space instead of letting a
single octave fill the buffer.

Three selection modes:

| Mode | Flags | Behavior |
|------|-------|----------|
| `score-per-octave` (default) | `use_score_filter=True`, `use_per_octave_cap=True` | Top `k` by sharpness, budget split per octave |
| `score-global` | `use_score_filter=True`, `use_per_octave_cap=False` | Top `k` by sharpness across the whole image |
| `legacy` | `use_score_filter=False` | Original upstream truncation |

### 3. Python wrapper (pybind11)

A `cudasift_py` module exposes extraction directly from NumPy, so the GPU
detector can be called from a Python pipeline without going through the C++
executable.

### Other changes

- CUDA architecture is parametrizable from CMake (`-DCMAKE_CUDA_ARCHITECTURES`),
  defaulting to `75`, instead of being hard coded.
- Console output is gated behind `VERBOSE`, so the library stays quiet when used
  as a Python module.

---

## Build

Requirements: CUDA toolkit, OpenCV, CMake, and pybind11 (`pip install pybind11`
or a system package).

```bash
mkdir build && cd build
# set the architecture of your GPU (e.g. 86 for Ampere, 75 for Turing)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
make -j
```

This builds:

- `cudasift` — the upstream demo executable.
- `cudasift_py` — the Python extension module (`cudasift_py*.so`).

Put the build directory on your `PYTHONPATH`, or copy the `.so` next to your
script, so `import cudasift_py` works.

---

## Python usage

```python
import numpy as np
import cudasift_py

# image: 2D float32, shape (H, W). Values are used as is (no auto scaling).
# For 0..1 inputs the default threshold is appropriate; for 0..255, raise it.
img = load_grayscale_float32("frame.png")  # your loader -> np.float32 (H, W)

keypoints, scales, oris, scores, descriptors = cudasift_py.extract(
    img,
    num_octaves=4,
    init_blur=1.0,
    thresh=0.4,
    lowest_scale=0.0,
    scale_up=False,
    max_pts=2048,
    dev_num=0,
    use_score_filter=True,    # top-k by sharpness (default)
    use_per_octave_cap=True,  # split the budget across octaves (default)
)

# keypoints:   (N, 2) float32  -> (x, y)
# scales:      (N,)   float32
# oris:        (N,)   float32  -> orientation
# scores:      (N,)   float32  -> extraction-time sharpness
# descriptors: (N, 128) float32
```

`extract()` arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `image` | required | 2D `float32` array, shape `(H, W)` |
| `num_octaves` | `4` | Number of octaves |
| `init_blur` | `1.0` | Initial Gaussian blur |
| `thresh` | `0.4` | DoG response threshold (raise for 0..255 inputs) |
| `lowest_scale` | `0.0` | Smallest scale to keep |
| `scale_up` | `False` | Upsample the image 2x before detection |
| `max_pts` | `2048` | Maximum keypoints returned (the final cap) |
| `dev_num` | `0` | CUDA device index |
| `use_score_filter` | `True` | Top-k by sharpness instead of legacy truncation |
| `use_per_octave_cap` | `True` | Split the keypoint budget evenly across octaves |

---

## C++ API changes

`InitSiftData` takes two new optional flags (both default `true`):

```cpp
void InitSiftData(SiftData &data, int num = 1024, bool host = false,
                  bool dev = true, bool useScoreFilter = true,
                  bool usePerOctaveCap = true);
```

`SiftData` gains a few fields. `maxPts` is now the **final returned cap**;
`maxWorkPts` is the internal candidate capacity allocated on the device:

```cpp
int  maxPts;          // final returned Sift point cap
int  maxWorkPts;      // internal candidate capacity (4 * maxPts when filtering)
bool useScoreFilter;  // post-detection top-k by abs(sharpness)
bool usePerOctaveCap; // split the budget evenly across octaves
int  numOctaves;      // used for the per-octave quota
```

Existing calls keep working: with the defaults you get score filtering with the
per octave cap; pass `useScoreFilter = false` to restore the original behavior.

---

## Credits and license

Forked from [CudaSift](https://github.com/Celebrandil/CudaSift) by Mårten
Björkman (Celebrandil). The original algorithm and the bulk of the CUDA code are
his. See `LICENSE` for the original license terms, which this fork keeps.
