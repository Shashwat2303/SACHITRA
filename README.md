# SACHITRA: Scalable Application of CPU-based HTML Integrated Transformer for Regenerative Augmented Image Generation Model

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![WebAssembly](https://img.shields.io/badge/WebAssembly-Enabled-orange.svg)
![Status](https://img.shields.io/badge/Status-Research--Ready-yellow.svg)

> **SACHITRA** (Sanskrit: सचित्र - "with image") is a groundbreaking browser-native generative AI framework that enables high-quality image, video, and presentation generation on commodity CPU hardware without GPU dependencies.

## 🎯 Key Innovations

| Component | Description | Key Benefit |
|-----------|-------------|-------------|
| **QHP** | Quantized HTML Pipeline | Browser-native execution with WebAssembly SIMD |
| **AFDT** | Adaptive Finite-Difference Trajectory | 2.1×-3.4× speedup with formal error bounds |
| **CHAD** | Cross-Layer Historical Attention Distillation | 16× memory reduction with <0.26% accuracy loss |

## 📊 Performance Highlights

```
┌────────────────────────────────────────────────────────────────┐
│  SACHITRA-50: 2.1× speedup | CLIP-IQA: 0.80 | 55.7s latency   │
│  SACHITRA-25: 3.4× speedup | CLIP-IQA: 0.78 | 34.8s latency   │
│  Target Hardware: Intel Core i5, 8GB RAM (No GPU required)   │
└────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Browser Demo (No Installation Required)
```bash
# Clone the repository
git clone https://github.com/sachitra/sachitra-framework.git
cd sachitra-framework/demos/image

# Open index.html in any modern browser
open index.html
```

### Python Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run image generation demo
python demos/image/generate.py --prompt "A serene mountain landscape at sunset" --steps 50

# Run benchmark suite
python benchmarks/run_benchmark.py --config configs/benchmark.yaml
```

## 📁 Repository Structure

```
sachitra-repository/
├── src/
│   ├── qhp/          # Quantized HTML Pipeline implementation
│   ├── afdt/         # Adaptive Finite-Difference Trajectory controller
│   ├── chad/         # Cross-Layer Historical Attention Distillation
│   ├── core/         # Core flow-matching inference engine
│   └── utils/        # Utilities and helper functions
├── demos/
│   ├── image/        # Image generation demo applications
│   ├── video/        # Video synthesis demo applications
│   └── ppt/          # Presentation generation demo
├── benchmarks/       # Performance benchmarking tools
├── verification/     # Formal verification and testing suite
├── docs/             # Documentation and technical specs
└── tests/            # Unit and integration tests
```

## 🔬 Research Paper

This repository accompanies the peer-reviewed research paper published at IEEE CAI 2026:

> **SACHITRA: Scalable Application of CPU-based HTML Integrated Transformer for Regenerative Augmented Image Generation Model**
> *Shashwat Pandey, Digital India BHASHINI Division, Ministry of Electronics & Information Technology, Government of India*

### Key Theorems

**Theorem 1 (AFDT Quantized Error Bound)**
$$
e_T \leq \underbrace{\frac{M e^{L_x}}{2} \cdot \frac{|S|}{T^3}}_{approximation} + \underbrace{e^{L_x} \cdot K \cdot \varepsilon_q}_{quantization}
$$

**Theorem 2 (CHAD Distortion Bound)**
$$
\|A_{cur}^L(\text{fp32}) - A_{cur}^L(\text{INT4})\| \leq \delta_q
$$

## 🧪 Verification & Testing

### Running the Verification Suite
```bash
# Run formal verification
python verification/formal_verify.py

# Execute benchmark tests
python benchmarks/run_benchmark.py --all

# Test CHAD attention mechanism
python tests/test_chad.py --dataset cifar100

# Validate AFDT error bounds
python tests/test_afdt.py --theorem-validation
```

### Expected Results

| Test | Metric | Target | Achieved |
|------|--------|--------|----------|
| AFDT Error Bound | $e_T$ convergence | $O(1/T^3)$ | Verified ✓ |
| CHAD Memory | Buffer size reduction | 16× | 16.1× ✓ |
| CLIP-IQA | Quality score | >0.78 | 0.80 ✓ |
| Latency | CPU generation time | <60s | 55.7s ✓ |

## 🖼️ Demo Capabilities

### Image Generation
```python
from src.core.sachitra_inference import SACHITRAInference

model = SACHITRAInference(config="configs/sachitra_50.json")
image = model.generate(
    prompt="A futuristic cityscape with flying vehicles",
    resolution=(512, 512),
    steps=50
)
image.save("output/futuristic_city.png")
```

### Video Generation
```python
from src.core.video_synthesizer import VideoSynthesizer

video_gen = VideoSynthesizer(model=model)
frames = video_gen.generate_sequence(
    prompts=["Frame 1 description", "Frame 2 description"],
    fps=24,
    duration=5
)
video_gen.assemble_video(frames, output="output/animation.mp4")
```

### Presentation Generation
```python
from src.core.ppt_generator import PPTGenerator

ppt_gen = PPTGenerator(model=model)
slides = ppt_gen.create_presentation(
    topic="AI Research Summary",
    slides=[
        {"title": "Introduction", "content": "...", "image": "auto"},
        {"title": "Methodology", "content": "...", "image": "auto"},
        {"title": "Results", "content": "...", "image": "auto"}
    ]
)
ppt_gen.export(slides, format="pptx", output="output/presentation.pptx")
```

## 🔧 Technical Specifications

### Hardware Requirements
- **Minimum**: Intel Core i5 (6 cores), 8GB RAM
- **Recommended**: Intel Core i7/i9 or AMD Ryzen 5/7
- **GPU**: Not required (CPU-only execution)

### Software Requirements
- Python 3.9+
- Chrome/Firefox/Safari (latest version)
- WebAssembly support

### Browser Compatibility
| Browser | Version | WASM SIMD | SharedArrayBuffer |
|---------|---------|------------|-------------------|
| Chrome  | 124+    | ✓          | ✓                 |
| Firefox | 120+    | ✓          | ✓                 |
| Safari  | 17+     | ✓          | ✓                 |
| Edge    | 124+    | ✓          | ✓                 |

## 📈 Benchmark Results

### GenEval Benchmark (553 prompts)

| Method | Overall | CLIP-IQA | Speedup | Latency |
|--------|---------|----------|---------|---------|
| Full FM (GPU, FP32) | 0.78 | 0.85 | 1.00× | 36.2s |
| CPU INT4 (baseline) | 0.73 | 0.80 | 0.31× | 118.4s |
| TeaCache (CPU, INT4) | 0.71 | 0.77 | 0.58× | 62.7s |
| AFDT only | 0.73 | 0.79 | 0.61× | 59.3s |
| **SACHITRA-50** | **0.74** | **0.80** | **0.65×** | **55.7s** |
| **SACHITRA-25** | **0.72** | **0.78** | **1.04×** | **34.8s** |

### Classification Validation (CHAD)

| Method | CIFAR-100 | TinyImageNet | Buffer Size | α |
|--------|-----------|--------------|-------------|-----|
| ViT Baseline (FP32) | 75.74% | 57.82% | N/A | — |
| HAViT-FP32 | 77.07% | 59.07% | $Bhn^2 \times 32b$ | 0.45 fixed |
| CHAD-INT4 (fixed) | 76.58% | 58.71% | $Bhn^2 \times 4b$ | 0.45 fixed |
| **CHAD-INT4 (dynamic)** | **76.81%** | **58.89%** | $Bhn^2 \times 4b$ | 0.35→0.55 |

## 📝 Citation

```bibtex
@article{pandey2026sachitra,
  title={SACHITRA: Scalable Application of CPU-based HTML Integrated Transformer for Regenerative Augmented Image Generation Model},
  author={Pandey, Shashwat},
  journal={2026 IEEE Conference on Artificial Intelligence (CAI)},
  year={2026},
  institution={Digital India BHASHINI Division, MeitY, Government of India}
}
```

## 📄 License

Released under the **Indian Open Government Licence 2.0** as a foundational contribution to India's indigenous AI infrastructure stack under the **Aatmanirbhar Bharat** programme.

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📧 Contact

- **Author**: Shashwat Pandey
- **Email**: shashwat.dibd@gmail.com
- **Institution**: Digital India BHASHINI Division, MeitY, Government of India

---

**Made with ❤️ in India for the world**
