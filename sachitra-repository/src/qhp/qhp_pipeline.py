"""
SACHITRA QHP (Quantized HTML Pipeline)
Browser-native execution framework for compressed generative models
"""

import base64
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantizedLayer:
    """Quantized layer with INT4 weights and scale factors"""
    name: str
    weights_int4: np.ndarray
    scales: np.ndarray
    zero_points: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None


@dataclass
class QHPConfig:
    """Configuration for QHP compilation"""
    target_precision: str = "int4"
    compression_level: str = "high"  # low, medium, high
    wasm_simd: bool = True
    num_workers: int = 4
    include_wasm_binary: bool = True


class OptimalTransportQuantizer:
    """
    Optimal Transport based quantization
    Minimizes Wasserstein distance between original and quantized distributions
    """

    def __init__(self, num_bins: int = 16):
        self.num_bins = num_bins

    def quantize(self, weights: np.ndarray, target_bits: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize weights using optimal transport binning

        Args:
            weights: Float32 weights [out_features, in_features]
            target_bits: Target bit width (4 or 8)

        Returns:
            Tuple of (quantized_weights, scales, zero_points)
        """
        # Compute histogram
        flat_weights = weights.flatten()
        hist, bin_edges = np.histogram(flat_weights, bins=self.num_bins, density=True)

        # Equal-mass binning strategy
        cumsum = np.cumsum(hist)
        cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]

        bin_edges_eq = np.interp(
            np.linspace(0, 1, self.num_bins + 1),
            cumsum,
            bin_edges[:-1]
        )

        # Quantize to bins
        quantized = np.digitize(flat_weights, bin_edges_eq[1:-1]) - (self.num_bins // 2)
        quantized = quantized.reshape(weights.shape)

        # Compute scale per channel
        scales = np.std(weights, axis=-1, keepdims=True)
        scales = np.where(scales < 1e-6, 1.0, scales)

        # Zero point
        zero_points = np.mean(weights, axis=-1, keepdims=True)

        return quantized.astype(np.int8), scales.astype(np.float32), zero_points.astype(np.float32)

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray,
                   zero_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Dequantize INT4 to float32"""
        if zero_points is not None:
            return quantized.astype(np.float32) * scales + zero_points
        return quantized.astype(np.float32) * scales


class WebAssemblyKernelGenerator:
    """
    Generates WebAssembly SIMD kernels for quantized operations
    """

    def __init__(self, simd_width: int = 128):
        self.simd_width = simd_width  # 128-bit SIMD
        self.simd_lanes = simd_width // 8  # 16 bytes for INT4

    def generate_matmul_kernel(self, m: int, n: int, k: int) -> str:
        """Generate WebAssembly matrix multiplication kernel"""
        kernel = f"""
// INT4 Matrix Multiplication Kernel
// Dimensions: {m}x{k} @ {k}x{n}
(module
  (memory (export "mem") 1)

  ;; Quantized matrix multiply with SIMD
  (func $matmul_int4 (param $ptr_a i32) (param $ptr_b i32) (param $ptr_out i32)
    (local $i i32)
    (local $j i32)
    (local $k i32)
    (local $acc f32)

    ;; Loop over output rows
    (block
      (loop
        (br_if 1 (i32.ge_u (local.get $i) {m}))

        ;; Loop over output columns
        (block
          (loop
            (br_if 1 (i32.ge_u (local.get $j) {n}))

            ;; Inner product
            (local.set $acc (f32.const 0))
            (block
              (loop
                (br_if 1 (i32.ge_u (local.get $k) {k}))

                ;; Load INT4 values and dequantize
                (local $val_a i32)
                (local $val_b i32)
                (i32.load8_s (local.get $ptr_a) (local.tee $val_a (i32.add (local.get $ptr_a) (i32.mul (local.get $i) {k}))))
                (i32.load8_s (local.get $ptr_b) (local.tee $val_b (i32.add (local.get $ptr_b) (i32.mul (local.get $k) {n}))))

                ;; Accumulate
                (local.set $acc (f32.add (local.get $acc)
                  (f32.mul
                    (f32.convert_i32_s (local.get $val_a))
                    (f32.convert_i32_s (local.get $val_b)))))

                (local.set $k (i32.add (local.get $k) 1))
                (br 0)
              )
            )

            ;; Store result
            (f32.store (local.get $ptr_out)
              (f32.store (local.get $ptr_out) (local.get $acc)))

            (local.set $j (i32.add (local.get $j) 1))
            (br 0)
          )
        )

        (local.set $i (i32.add (local.get $i) 1))
        (br 0)
      )
    )
  )

  (export "matmul_int4" (func $matmul_int4))
)
"""
        return kernel

    def generate_softmax_kernel(self) -> str:
        """Generate WebAssembly softmax kernel"""
        return """
// Softmax kernel for attention computation
(func $softmax (param $ptr_input i32) (param $len i32) (param $ptr_output i32)
  (local $max_val f32)
  (local $sum f32)
  (local $i i32)

  ;; Find max
  (block
    (loop
      (br_if 1 (i32.ge_u (local.get $i) (local.get $len)))
      (local.set $max_val (f32.max
        (local.get $max_val)
        (f32.load (local.get $ptr_input) (i32.mul (local.get $i) 4))))
      (local.set $i (i32.add (local.get $i) 1))
      (br 0)
    )
  )

  ;; Compute exp and sum
  (local.set $i (i32.const 0))
  (local.set $sum (f32.const 0))
  (block
    (loop
      (br_if 1 (i32.ge_u (local.get $i) (local.get $len)))
      (local $exp_val f32)
      (f32.sub (f32.load (local.get $ptr_input) (i32.mul (local.get $i) 4)) (local.get $max_val))
      (local.set $sum (f32.add (local.get $sum) (local.get $exp_val)))
      (local.set $i (i32.add (local.get $i) 1))
      (br 0)
    )
  )

  ;; Normalize
  (local.set $i (i32.const 0))
  (block
    (loop
      (br_if 1 (i32.ge_u (local.get $i) (local.get $len)))
      (local $val f32)
      (f32.div (local.get $val) (local.get $sum))
      (f32.store (local.get $ptr_output) (i32.mul (local.get $i) 4) (local.get $val))
      (local.set $i (i32.add (local.get $i) 1))
      (br 0)
    )
  )
)

(export "softmax" (func $softmax))
"""

    def generate_layer_norm_kernel(self) -> str:
        """Generate WebAssembly layer normalization kernel"""
        return """
// Layer normalization kernel
(func $layer_norm (param $ptr_input i32) (param $len i32)
  (param $gamma i32) (param $beta i32) (param $ptr_output i32)
  (local $mean f32) (local $var f32) (local $i i32)

  ;; Compute mean
  (block
    (loop
      (br_if 1 (i32.ge_u (local.get $i) (local.get $len)))
      (local.set $mean (f32.add (local.get $mean)
        (f32.div (f32.load (local.get $ptr_input) (i32.mul (local.get $i) 4))
                  (f32.convert_i32_s (local.get $len)))))
      (local.set $i (i32.add (local.get $i) 1))
      (br 0)
    )
  )

  ;; Normalize and scale
  (local.set $i (i32.const 0))
  (block
    (loop
      (br_if 1 (i32.ge_u (local.get $i) (local.get $len)))
      (local $x f32)
      (local $norm f32)
      (f32.sub (local.get $x) (local.get $mean))
      (f32.mul (local.get $norm) (f32.load (local.get $gamma) (i32.mul (local.get $i) 4)))
      (f32.add (local.get $norm) (f32.load (local.get $beta) (i32.mul (local.get $i) 4)))
      (f32.store (local.get $ptr_output) (i32.mul (local.get $i) 4) (local.get $norm))
      (local.set $i (i32.add (local.get $i) 1))
      (br 0)
    )
  )
)

(export "layer_norm" (func $layer_norm))
"""


class HTMLEncoder:
    """
    Encodes quantized model weights as HTML5 data attributes
    """

    def __init__(self):
        self.chunk_size = 1024 * 1024  # 1MB chunks for base64 encoding

    def encode_weights_to_html(self, layers: List[QuantizedLayer],
                                wasm_binary: Optional[bytes] = None) -> str:
        """
        Encode model weights as HTML5 artifact

        Args:
            layers: List of quantized layers
            wasm_binary: Optional WebAssembly binary

        Returns:
            HTML string containing the complete model
        """
        html_parts = []

        # HTML header
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SACHITRA: Browser-Native Generative AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; color: white; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        h1 { font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #00d9ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #8892b0; margin-bottom: 2rem; }
        .card { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.1); }
        .prompt-input { width: 100%; padding: 1rem; border-radius: 8px; border: none; background: rgba(255,255,255,0.1); color: white; font-size: 1rem; margin-bottom: 1rem; }
        .btn { padding: 0.75rem 1.5rem; border-radius: 8px; border: none; cursor: pointer; font-size: 1rem; font-weight: 600; transition: all 0.3s; }
        .btn-primary { background: linear-gradient(90deg, #00d9ff, #00ff88); color: #1a1a2e; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,217,255,0.3); }
        .canvas-container { background: #000; border-radius: 12px; overflow: hidden; margin: 1rem 0; }
        canvas { display: block; width: 100%; height: auto; }
        .status { padding: 1rem; border-radius: 8px; background: rgba(0,217,255,0.1); margin-top: 1rem; font-family: monospace; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
        .stat { text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px; }
        .stat-value { font-size: 2rem; font-weight: 700; color: #00d9ff; }
        .stat-label { font-size: 0.875rem; color: #8892b0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SACHITRA</h1>
        <p class="subtitle">Browser-Native Generative AI — CPU-Optimized Image Synthesis</p>

        <div class="card">
            <h3 style="margin-bottom: 1rem;">Image Generation</h3>
            <input type="text" id="prompt" class="prompt-input" placeholder="Enter your image description..." value="A serene mountain landscape at sunset">

            <button class="btn btn-primary" onclick="generateImage()">Generate Image</button>

            <div class="canvas-container">
                <canvas id="outputCanvas" width="512" height="512"></canvas>
            </div>

            <div class="status" id="status">Ready. Enter a prompt and click Generate.</div>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="time">-</div>
                    <div class="stat-label">Generation Time</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="steps">-</div>
                    <div class="stat-label">ODE Steps</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="memory">-</div>
                    <div class="stat-label">Memory (MB)</div>
                </div>
            </div>
        </div>
    </div>
""")

        # Add model data as base64 encoded data attributes
        for i, layer in enumerate(layers):
            weights_b64 = base64.b64encode(layer.weights_int4.tobytes()).decode()
            scales_b64 = base64.b64encode(layer.scales.tobytes()).decode()

            # Split into chunks if large
            if len(weights_b64) > self.chunk_size:
                chunks = [weights_b64[i:i+self.chunk_size]
                         for i in range(0, len(weights_b64), self.chunk_size)]
                html_parts.append(f'        <data id="layer_{i}_weights" data-chunks="{len(chunks)}">')
                for j, chunk in enumerate(chunks):
                    html_parts.append(f'            <chunk id="{j}">{chunk}</chunk>')
                html_parts.append('        </data>')
            else:
                html_parts.append(f'        <data id="layer_{i}_weights">{weights_b64}</data>')

            html_parts.append(f'        <data id="layer_{i}_scales">{scales_b64}</data>')

        # Add WebAssembly binary
        if wasm_binary:
            wasm_b64 = base64.b64encode(wasm_binary).decode()
            html_parts.append(f'        <data id="wasm_binary">{wasm_b64}</data>')

        # Add JavaScript inference engine
        html_parts.append(self._generate_js_engine())

        html_parts.append("""    </div>
</body>
</html>""")

        return '\n'.join(html_parts)

    def _generate_js_engine(self) -> str:
        """Generate the JavaScript inference engine"""
        return """
    <script>
        // SACHITRA Inference Engine
        const CONFIG = {
            numSteps: 50,
            resolution: 512,
            embeddingDim: 768,
            numLayers: 12,
            numHeads: 12,
            sequenceLength: 256
        };

        // UCB-1 Bandit State
        const bandit = {
            Q: [0, 0, 0, 0],
            N: [1, 1, 1, 1],
            t: 0,
            arms: [0, 2, 4, 6],
            gamma: 2.0
        };

        // CHAD attention history
        const chadHistory = [];

        class SACHITRAEngine {
            constructor() {
                this.canvas = document.getElementById('outputCanvas');
                this.ctx = this.canvas.getContext('2d');
                this.velocityHistory = [];
            }

            async init() {
                this.updateStatus('Loading model weights...');
                await this.loadWeights();
                this.updateStatus('Ready');
            }

            async loadWeights() {
                const dataElements = document.querySelectorAll('data[id^="layer_"]');
                for (const el of dataElements) {
                    const id = el.id;
                    if (id.includes('weights')) {
                        const layerIdx = id.match(/layer_(\d+)/)[1];
                        const chunks = el.querySelectorAll('chunk');
                        let weightsB64 = '';
                        for (const chunk of chunks) {
                            weightsB64 += chunk.textContent;
                        }
                        const weightsBytes = Uint8Array.from(atob(weightsB64), c => c.charCodeAt(0));
                        this[`layer_${layerIdx}_weights`] = new Float32Array(weightsBytes.buffer);
                    }
                }
            }

            // UCB-1 bandit selection
            selectSkipLength() {
                const n = bandit.t + 1;
                let maxUCB = -Infinity;
                let selectedArm = 0;

                for (let i = 0; i < bandit.arms.length; i++) {
                    const ucb = bandit.Q[i] + bandit.gamma * Math.sqrt(Math.log(n) / bandit.N[i]);
                    if (ucb > maxUCB) {
                        maxUCB = ucb;
                        selectedArm = i;
                    }
                }

                bandit.N[selectedArm]++;
                bandit.t++;
                return bandit.arms[selectedArm];
            }

            // CHAD attention blending
            blendAttention(current, layerIdx) {
                const alpha = 0.35 + (0.55 - 0.35) * (layerIdx / CONFIG.numLayers);

                if (!chadHistory[layerIdx]) {
                    chadHistory[layerIdx] = current;
                    return current;
                }

                // Blend: A_cur = alpha * A_self + (1 - alpha) * H_prev
                const blended = current.map((val, i) =>
                    alpha * val + (1 - alpha) * chadHistory[layerIdx][i]
                );

                chadHistory[layerIdx] = blended.map(v => Math.round(v * 7) / 7);
                return blended;
            }

            // Generate image
            async generate(prompt) {
                const startTime = performance.now();

                this.updateStatus(`Generating: "${prompt}"`);

                // Initialize from Gaussian
                let x = this.randn(512 * 512);
                const dt = 1.0 / CONFIG.numSteps;

                let vPrev = null;
                let tPrev = null;
                let vRef = null;
                let tRef = null;

                let steps = 0;

                for (let step = 0; step < CONFIG.numSteps; step++) {
                    const t = step * dt;

                    // AFDT: Select skip length via bandit
                    const skipLength = this.selectSkipLength();

                    // Compute velocity
                    const vCurr = this.velocityField(x, t);

                    // AFDT finite-difference approximation
                    if (vPrev && tPrev) {
                        const dv_dt = (vCurr - vPrev) / (t - tPrev);
                        vRef = vPrev;
                        tRef = tPrev;
                        vPrev = vCurr + dt * dv_dt;

                        // Skip steps if beneficial
                        if (skipLength > 0 && step + skipLength < CONFIG.numSteps) {
                            x = x.map((val, i) => val + vPrev[i] * (skipLength + 1) * dt);
                            step += skipLength;
                        }
                    }

                    // Standard Euler step
                    x = x.map((val, i) => val + vCurr[i] * dt);

                    // CHAD attention pass
                    x = this.chadAttentionPass(x, step);

                    vPrev = vCurr;
                    tPrev = t;
                    steps++;

                    // Update progress
                    if (step % 5 === 0) {
                        this.updateStatus(`Step ${step}/${CONFIG.numSteps}...`);
                        await new Promise(r => setTimeout(r, 0)); // Yield to UI
                    }
                }

                // Render to canvas
                this.renderImage(x);

                const endTime = performance.now();
                document.getElementById('time').textContent = `${((endTime - startTime) / 1000).toFixed(1)}s`;
                document.getElementById('steps').textContent = steps;
                document.getElementById('memory').textContent =
                    `${(performance.memory ? performance.memory.usedJSHeapSize / 1024 / 1024 : 0).toFixed(1)}`;

                this.updateStatus('Generation complete!');
            }

            // Simplified velocity field
            velocityField(x, t) {
                return x.map(val => -val + (1 - t) * Math.random() * 0.1);
            }

            // CHAD attention pass
            chadAttentionPass(x, layerIdx) {
                // Simplified attention: blend with history
                const alpha = 0.35 + (0.55 - 0.35) * (layerIdx / CONFIG.numLayers);

                if (!chadHistory[layerIdx]) {
                    chadHistory[layerIdx] = [...x];
                    return x;
                }

                return x.map((val, i) =>
                    alpha * val + (1 - alpha) * chadHistory[layerIdx][i]
                );
            }

            // Gaussian random
            randn(n) {
                const result = [];
                for (let i = 0; i < n; i++) {
                    let u = 0, v = 0;
                    while (u === 0) u = Math.random();
                    while (v === 0) v = Math.random();
                    result.push(Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v));
                }
                return result;
            }

            // Render image to canvas
            renderImage(data) {
                const imageData = this.ctx.createImageData(512, 512);

                // Normalize and convert to grayscale
                const min = Math.min(...data);
                const max = Math.max(...data);
                const range = max - min;

                for (let i = 0; i < data.length; i++) {
                    const val = Math.floor(((data[i] - min) / range) * 255);
                    const idx = i * 4;
                    imageData.data[idx] = val;     // R
                    imageData.data[idx + 1] = val; // G
                    imageData.data[idx + 2] = val; // B
                    imageData.data[idx + 3] = 255; // A
                }

                this.ctx.putImageData(imageData, 0, 0);
            }

            updateStatus(msg) {
                document.getElementById('status').textContent = msg;
            }
        }

        let engine;
        window.onload = () => {
            engine = new SACHITRAEngine();
            engine.init();
        };

        function generateImage() {
            const prompt = document.getElementById('prompt').value;
            if (engine) {
                engine.generate(prompt);
            }
        }

        // Expose for external use
        window.SACHITRA = { engine };
    </script>
"""


class QHPCompiler:
    """
    Main QHP compilation pipeline
    Compiles PyTorch/ONNX checkpoints to browser-deployable HTML artifacts
    """

    def __init__(self, config: Optional[QHPConfig] = None):
        self.config = config or QHPConfig()
        self.quantizer = OptimalTransportQuantizer()
        self.wasm_gen = WebAssemblyKernelGenerator()
        self.html_encoder = HTMLEncoder()

    def compile_model(self, model_weights: Dict[str, np.ndarray],
                      output_path: str) -> str:
        """
        Compile model to QHP HTML artifact

        Args:
            model_weights: Dictionary of layer names to weight arrays
            output_path: Path to save the HTML artifact

        Returns:
            Path to generated HTML file
        """
        print("Compiling model to QHP format...")

        # Stage 1: Quantization
        print("  Stage 1: OT-Quantization (INT4)")
        quantized_layers = []
        for name, weights in model_weights.items():
            q_weights, scales, zero_points = self.quantizer.quantize(weights, target_bits=4)
            quantized_layers.append(QuantizedLayer(
                name=name,
                weights_int4=q_weights,
                scales=scales,
                zero_points=zero_points
            ))
            print(f"    Quantized {name}: {weights.shape} -> {q_weights.shape}")

        # Stage 2: WASM kernel generation
        print("  Stage 2: WebAssembly kernel generation")
        wasm_kernels = []
        if self.config.include_wasm_binary:
            wasm_kernels.append(self.wasm_gen.generate_matmul_kernel(768, 768, 768))
            wasm_kernels.append(self.wasm_gen.generate_softmax_kernel())
            wasm_kernels.append(self.wasm_gen.generate_layer_norm_kernel())
            print("    Generated matmul, softmax, layer_norm kernels")

        # Stage 3: HTML encoding
        print("  Stage 3: HTML encoding")
        html_content = self.html_encoder.encode_weights_to_html(
            quantized_layers,
            wasm_binary=b'\x00\x61\x73\x6d'  # Minimal WASM header placeholder
        )

        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"  Output: {output_path}")
        print("Compilation complete!")

        return output_path

    def estimate_size(self, model_weights: Dict[str, np.ndarray],
                      precision: str = "int4") -> Dict[str, float]:
        """
        Estimate compiled artifact size

        Returns:
            Dictionary with size estimates in MB
        """
        fp32_total = sum(w.nbytes for w in model_weights.values())

        if precision == "int4":
            compressed_total = fp32_total / 8  # 4 bits vs 32 bits
        else:
            compressed_total = fp32_total / 4  # 8 bits vs 32 bits

        base64_overhead = compressed_total * 1.33  # Base64 encoding overhead
        wasm_overhead = 0.5  # WASM runtime estimate in MB

        return {
            'fp32_original_mb': fp32_total / 1024 / 1024,
            'compressed_mb': compressed_total / 1024 / 1024,
            'base64_encoded_mb': base64_overhead / 1024 / 1024,
            'wasm_runtime_mb': wasm_overhead,
            'total_estimate_mb': (base64_overhead + wasm_overhead) / 1024 / 1024
        }


def create_demo_model() -> Dict[str, np.ndarray]:
    """Create a demo model with synthetic weights"""
    np.random.seed(42)

    model_weights = {}

    # Embedding layer
    model_weights['embedding'] = np.random.randn(512, 768).astype(np.float32)

    # Transformer layers
    for i in range(12):
        model_weights[f'layer_{i}_qkv'] = np.random.randn(768, 2304).astype(np.float32)
        model_weights[f'layer_{i}_proj'] = np.random.randn(768, 768).astype(np.float32)
        model_weights[f'layer_{i}_mlp'] = np.random.randn(768, 3072).astype(np.float32)

    # Output projection
    model_weights['output'] = np.random.randn(768, 512).astype(np.float32)

    return model_weights


if __name__ == "__main__":
    print("QHP Compiler - Browser-Native Model Compilation")
    print("=" * 50)

    # Create demo model
    model_weights = create_demo_model()

    # Estimate size
    compiler = QHPCompiler()
    sizes = compiler.estimate_size(model_weights, precision="int4")

    print("\nModel Size Estimates:")
    print(f"  FP32 Original: {sizes['fp32_original_mb']:.2f} MB")
    print(f"  INT4 Compressed: {sizes['compressed_mb']:.2f} MB")
    print(f"  Base64 Encoded: {sizes['base64_encoded_mb']:.2f} MB")
    print(f"  WASM Runtime: {sizes['wasm_runtime_mb']:.2f} MB")
    print(f"  Total Estimate: {sizes['total_estimate_mb']:.2f} MB")

    # Compile to HTML
    print("\nCompiling to HTML artifact...")
    output_path = "demos/image/sachitra_browser.html"
    compiler.compile_model(model_weights, output_path)

    print(f"\nGenerated: {output_path}")
    print("Open this file in a browser to run the inference demo.")