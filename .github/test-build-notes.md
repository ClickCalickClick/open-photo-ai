**Unofficial test build** of Open Photo AI with **WebGPU GPU acceleration for AMD and Intel GPUs**. It is not a vegidio/open-photo-ai release; it exists so people can test the upcoming WebGPU pull request ([discussion in #46](https://github.com/vegidio/open-photo-ai/issues/46)).

Version `{{VERSION}}`, built by GitHub Actions from commit `{{COMMIT}}` with the upstream project's own build steps. Telemetry and analytics are **off** in this build.

### What's in it
Upstream `main` plus:
- **WebGPU processor**: runs the AI models on AMD/Intel GPUs (Vulkan on Linux, Direct3D 12 on Windows). Auto picks it when there is no NVIDIA/Apple processor; you can also choose it in **Settings → AI Processor**. It downloads a ~2–3 MB plugin on first use.
- Memory budget fix for integrated GPUs (vegidio/open-photo-ai#53).
- The Open dialog shows upper-case extensions on Linux, e.g. `.DNG`, `.CR2` (#51).
- Opens lossy DNGs, including Lightroom Smart Previews (#54), and JPEG XL DNGs, including DxO PureRAW 6 output (#52).

### Testing
1. Unzip and run. On Linux, WebGPU needs Vulkan: `vulkaninfo --summary` should list your GPU (install `vulkan-radeon`/`mesa-vulkan-drivers` or your distro's equivalent).
2. Process a photo with **Auto** or **WebGPU**, then the same photo with **CPU**, and note both times.
3. Please post in #46: your GPU and OS, the two times, whether the results look the same, and anything that went wrong. Attach the log: `~/.config/open-photo-ai/logs/opai.log` on Linux, `%AppData%\open-photo-ai\logs\opai.log` on Windows.

Known limits: WebGPU runs the HD (fp32) models; SD (fp16) falls back to the CPU. Colorization runs part of its model on the CPU. The builds are unsigned, so Windows SmartScreen will warn ("More info → Run anyway").

### SHA-256
```
