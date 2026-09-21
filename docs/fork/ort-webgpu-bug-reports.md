# Draft bug reports for microsoft/onnxruntime (WebGPU plugin EP)

Two defects in the ONNX Runtime WebGPU plugin execution provider found while adding the WebGPU processor to Open
Photo AI (branch `webgpu-ep`). The app works around both; these reports are what would let the workarounds go. Not
yet filed.

Workarounds in the branch: `models/colorization/jaipur` pins the two affected convolutions to the CPU via
`forceCpuNodeNames`; `internal/utils/webgpu.go` routes every fp16 graph away from the provider; `models/lightadjustment/paris`
pins its `Pow` node to the CPU (see report 3).

---

## 1. GPU hang: chained non-vec4 Conv at large spatial size (RADV)

**Title:** [WebGPU EP] Two chained 3x3 Conv (C=303, 560x560, fp32) hang the AMD GPU on Linux/Vulkan (RADV); works at C=304 or 280x280

**Environment**
- onnxruntime 1.26.0 + onnxruntime-ep-webgpu 0.1.0, and onnxruntime 1.30.0 + onnxruntime-ep-webgpu 0.3.0 (both reproduce)
- Linux (CachyOS, kernel 7.2.6), Mesa 26.2.2 RADV, AMD Radeon 680M (gfx1035, Rembrandt iGPU), 14 GiB RAM, 1 GiB VRAM carve-out / 7.2 GiB GTT
- VK_DRIVER_FILES pinned to radeon_icd.json (lavapipe excluded)

**Repro** (python, synthetic model, no weights needed):
```python
import numpy as np, onnx, onnxruntime as ort, onnxruntime_ep_webgpu as w
from onnx import helper, TensorProto
C = 303  # hangs; 304 works
rng = np.random.default_rng(0); inits=[]; nodes=[]; prev="input"
for i in range(2):
    inits += [helper.make_tensor(f"w{i}", TensorProto.FLOAT, [C,C,3,3], (rng.standard_normal((C,C,3,3))*0.01).astype(np.float32).ravel()),
              helper.make_tensor(f"b{i}", TensorProto.FLOAT, [C], rng.standard_normal(C).astype(np.float32))]
    nodes += [helper.make_node("Conv", [prev,f"w{i}",f"b{i}"], [f"c{i}"], kernel_shape=[3,3], pads=[1,1,1,1]),
              helper.make_node("Relu", [f"c{i}"], [f"r{i}"])]; prev=f"r{i}"
nodes[-1].output[0] = "out"
g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,C,"H","W"])],
                      [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1,C,"H","W"])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("",18)]); m.ir_version = 9; onnx.save(m, "chain2.onnx")
ort.register_execution_provider_library("webgpu", w.get_library_path())
dev = next(d for d in ort.get_ep_devices() if d.ep_name == w.get_ep_name())
so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.add_provider_for_devices([dev], {})
s = ort.InferenceSession("chain2.onnx", sess_options=so)
x = rng.random((1,C,560,560), dtype=np.float32)
print(s.run(None, {"input": x})[0].sum())   # C=303: dmesg "amdgpu: ring gfx_0.0.0 timeout" -> ring reset -> VK_ERROR_DEVICE_LOST / garbage (sum 0)
```

**Observed**
- C=303 @ 560x560: after ~2.5 s the kernel logs `amdgpu 0000:e4:00.0: ring gfx_0.0.0 timeout, signaled seq=..., emitted seq=...`, resets the ring; RADV reports `vkQueueSubmit() failed (VK_ERROR_DEVICE_LOST)`; ORT aborts in `BufferManager::Download` ("[Device] is lost") or returns zeros. Repeated hangs eventually leave the GPU in a reset loop needing a reboot.
- Same graph with a single Conv+Relu: fine (2.2 s). Two chained at 280x280: fine (1.0 s). Two chained at C=304 @ 560x560: fine (2.0 s).
- Not affected by storageBufferCacheMode (default/disabled/lazyRelease) or preferredLayout=NHWC. fp16 export of the real model also hangs.
- Real-world model: DeOldify-derived colorizer (vegidio/open-photo-ai `cl_jaipur`), decoder has two 303-ch 3x3 convs at 560x560. Forcing those two nodes to CPU via forceCpuNodeNames avoids the hang.

**Hypothesis:** the non-vectorized Conv path (in-channels % 4 != 0) with a ~380 MB GPU-resident input buffer (second conv reads the first's output) dispatches something the driver considers hung (>10 s lockup timeout, or a bad dispatch size). The vec4 path (C=304) with the same buffer sizes is fine.

Separate observation (may be its own issue): fp16 models return visibly wrong results on this EP (LayerNorm-heavy graphs: 80–90% of pixels off vs fp32 CPU, max abs err 0.3–1.4 on [0,1] images) while the CPU EP's fp16 run matches fp32 within 3e-4. Reproducible with vegidio/open-photo-ai `dn_gothenburg_fp16` / `sh_moscow_fp16`.


---

## 2. Wrong results for fp16 graphs

**Title:** [WebGPU EP] fp16 models return wrong results on Linux/Vulkan (RADV) while the same graphs match fp32 on the CPU EP

Same environment as above. Measured against the fp32 CPU result on a smooth synthetic 256x256 image in [0,1] (values are
absolute error; one 8-bit step is 0.0039):

| model (vegidio/open-photo-ai) | CPU fp16 max err | WebGPU fp16 max / mean err | pixels off by >1 step |
|---|---|---|---|
| dn_gothenburg_fp16 (denoise, LayerNorm-heavy) | 0.0003 | 0.317 / 0.017 | 79% |
| sh_moscow_fp16 (sharpen, LayerNorm-heavy) | 0.0001 | 1.365 / 0.069 | 92% |
| up_kyoto_2x_fp16 (conv upscaler) | 0.0008 | 0.017 / 0.002 | 14% |
| up_saitama_4x_fp16 (conv upscaler) | 0.0027 | 0.014 / 0.001 | 6% |

The fp32 exports of the same models match the CPU to <1e-4 on WebGPU. The LayerNorm-heavy graphs express the norm as
ReduceMean/Sub/Pow/ReduceSum/Sqrt/Div, which suggests fp16 overflow in the variance path (the CPU EP computes those in
fp32 internally). Reproduces with plugin 0.1.0 + ORT 1.26.0 and plugin 0.3.0 + ORT 1.30.0. Models are public on
https://huggingface.co/vegidio/open-photo-ai/tree/main/models.

---

## 3. Shader compile failure: Pow with a tensor exponent

**Title:** [WebGPU EP] Pow fails with "Invalid ShaderModule" when the exponent is a broadcast tensor

`la_paris_fp32.onnx` (same model zoo, 286 KB) ends with `Pow(image[1,3,1024,1024], gamma)` where `gamma` is a small
tensor the network computed, broadcast over the image. Session build succeeds; the first Run fails:

```
sequential_executor.cc:615 ExecuteKernel] Non-zero status code returned while running Pow node. Name:'node_pow_1'
Status Message: Failed to create a WebGPU compute pipeline: [Invalid ShaderModule "Pow"] is invalid due to a previous error.
```

Forcing the node to the CPU (`forceCpuNodeNames`) works. Plugin 0.1.0 + ORT 1.26.0, RADV as above. A minimal
synthetic repro (Pow of a [1,3,H,W] float input by a [1,3,1,1] float input) is the next step; not yet reduced.
