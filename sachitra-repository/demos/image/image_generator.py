"""
SACHITRA Image Generation Demo
Browser-native image synthesis using QHP, AFDT, and CHAD
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import SACHITRA modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.sachitra_inference import SACHITRAInference, SACHITRAConfig
from src.qhp.qhp_pipeline import QHPCompiler, create_demo_model
from src.afdt.afdt_controller import AFDTController, AFDTConfig
from src.chad.chad_module import CHADModule, CHADConfig


@dataclass
class GenerationResult:
    """Result of image generation"""
    image: np.ndarray
    prompt: str
    generation_time: float
    steps_used: int
    config: Dict[str, Any]
    metrics: Dict[str, float]


class ImageGenerator:
    """
    High-level image generation interface

    Integrates all SACHITRA components for browser-native image synthesis.
    """

    def __init__(self, model_size: str = "small", device: str = "browser"):
        self.model_size = model_size
        self.device = device

        # Load model
        self.config = SACHITRAConfig(
            num_steps=50,
            resolution=(512, 512),
            afdt_enabled=True,
            chad_enabled=True
        )

        self.model = SACHITRAInference(self.config)

        # Statistics
        self.total_generations = 0
        self.total_time = 0

    def generate(self, prompt: str, num_steps: Optional[int] = None,
                 seed: Optional[int] = None) -> GenerationResult:
        """
        Generate image from text prompt

        Args:
            prompt: Text description
            num_steps: Number of ODE integration steps
            seed: Random seed

        Returns:
            GenerationResult with image and metrics
        """
        import time

        start_time = time.time()

        # Generate
        if seed is None:
            seed = self.total_generations

        image = self.model.generate(prompt, num_steps, seed)

        generation_time = time.time() - start_time

        # Update statistics
        self.total_generations += 1
        self.total_time += generation_time

        # Compute metrics (simplified)
        metrics = {
            'mean_pixel': float(np.mean(image)),
            'std_pixel': float(np.std(image)),
            'min_pixel': float(np.min(image)),
            'max_pixel': float(np.max(image))
        }

        return GenerationResult(
            image=image,
            prompt=prompt,
            generation_time=generation_time,
            steps_used=num_steps or self.config.num_steps,
            config={
                'model_size': self.model_size,
                'device': self.device,
                'num_steps': self.steps_used if 'steps_used' in dir() else self.config.num_steps
            },
            metrics=metrics
        )

    def save_image(self, result: GenerationResult, path: str) -> None:
        """Save generated image to file"""
        import matplotlib.pyplot as plt

        # Normalize to [0, 255]
        img = (result.image - result.image.min()) / (result.image.max() - result.image.min())
        img = (img * 255).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Generated: "{result.prompt}"')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()

    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        return {
            'total_generations': self.total_generations,
            'total_time': self.total_time,
            'mean_time': self.total_time / max(self.total_generations, 1)
        }


class BrowserImageGenerator:
    """
    Browser-based image generator using QHP HTML artifact
    """

    def __init__(self, html_path: str):
        self.html_path = html_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from HTML"""
        # Configuration embedded in HTML
        return {
            'num_steps': 50,
            'resolution': 512,
            'embedding_dim': 768
        }

    def generate_html(self, output_path: str) -> str:
        """Generate standalone HTML artifact"""
        # Compile demo model
        model_weights = create_demo_model()
        compiler = QHPCompiler()

        # Generate HTML
        output_path = compiler.compile_model(model_weights, output_path)

        return output_path


def run_demo(prompts: list = None):
    """
    Run image generation demo

    Args:
        prompts: List of text prompts
    """
    if prompts is None:
        prompts = [
            "A serene mountain landscape at sunset",
            "A futuristic city with flying vehicles",
            "An abstract geometric composition",
            "A peaceful lake scene with mountains",
            "A vibrant sunset over the ocean"
        ]

    print("SACHITRA Image Generation Demo")
    print("=" * 60)

    # Create generator
    generator = ImageGenerator(model_size="small")

    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: '{prompt}'")

        result = generator.generate(prompt, num_steps=50, seed=i)

        print(f"    Time: {result.generation_time:.2f}s")
        print(f"    Steps: {result.steps_used}")
        print(f"    Mean pixel: {result.metrics['mean_pixel']:.4f}")

        # Save image
        output_dir = Path(__file__).parent
        output_path = output_dir / f"output_{i+1}.png"
        generator.save_image(result, str(output_path))
        print(f"    Saved: {output_path}")

    # Statistics
    print("\n" + "=" * 60)
    print("Generation Statistics:")
    stats = generator.get_statistics()
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Mean time: {stats['mean_time']:.2f}s")

    return generator


def generate_browser_demo(output_path: str = "sachitra_demo.html"):
    """
    Generate standalone browser demo HTML

    Args:
        output_path: Path to save HTML artifact
    """
    print("Generating browser demo...")

    generator = BrowserImageGenerator("")
    html_path = generator.generate_html(output_path)

    print(f"Generated: {html_path}")
    print("Open this file in a browser to run the demo.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SACHITRA Image Generation Demo")
    parser.add_argument("--prompts", nargs="+", help="Text prompts for generation")
    parser.add_argument("--browser", action="store_true", help="Generate browser demo HTML")
    parser.add_argument("--output", default="sachitra_demo.html", help="Output path")

    args = parser.parse_args()

    if args.browser:
        generate_browser_demo(args.output)
    else:
        run_demo(args.prompts)