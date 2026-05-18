"""
SACHITRA Video Generation Demo
Frame-by-frame video synthesis using SACHITRA
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.sachitra_inference import SACHITRAInference, SACHITRAConfig


@dataclass
class VideoConfig:
    """Configuration for video generation"""
    fps: int = 24
    duration_seconds: float = 5.0
    resolution: Tuple[int, int] = (512, 512)
    interpolation_steps: int = 10
    smooth_frames: int = 3


class VideoSynthesizer:
    """
    Video synthesis using SACHITRA framework

    Generates video frames by interpolating between keyframe prompts.
    """

    def __init__(self, model: Optional[SACHITRAInference] = None,
                 config: Optional[VideoConfig] = None):
        self.model = model or SACHITRAInference()
        self.config = config or VideoConfig()

        self.num_frames = int(self.config.fps * self.config.duration_seconds)
        self.frames_generated = 0

    def generate_frame(self, prompt: str, frame_idx: int) -> np.ndarray:
        """
        Generate a single frame

        Args:
            prompt: Frame description
            frame_idx: Frame index

        Returns:
            Frame as numpy array
        """
        # Add temporal context to prompt
        temporal_prompt = f"{prompt} | frame {frame_idx}/{self.num_frames}"

        # Generate with reduced steps for video
        image = self.model.generate(
            temporal_prompt,
            num_steps=25,  # Reduced for video speed
            seed=frame_idx
        )

        self.frames_generated += 1
        return image

    def interpolate_prompts(self, prompt_a: str, prompt_b: str,
                            num_intermediate: int = 10) -> List[str]:
        """
        Generate interpolated prompts between two keyframes

        Args:
            prompt_a: Start prompt
            prompt_b: End prompt
            num_intermediate: Number of intermediate prompts

        Returns:
            List of interpolated prompts
        """
        prompts = []

        for i in range(num_intermediate + 2):
            t = i / (num_intermediate + 1)

            # Simple linear interpolation description
            # In production, could use more sophisticated methods
            if i == 0:
                prompts.append(prompt_a)
            elif i == num_intermediate + 1:
                prompts.append(prompt_b)
            else:
                # Blend descriptions
                prompts.append(f"Transition: {prompt_a} to {prompt_b}")

        return prompts

    def generate_sequence(self, keyframe_prompts: List[str],
                          keyframe_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate video sequence from keyframe prompts

        Args:
            keyframe_prompts: List of keyframe descriptions
            keyframe_indices: Optional indices for keyframes

        Returns:
            List of generated frames
        """
        if keyframe_indices is None:
            # Evenly distribute keyframes
            keyframe_indices = np.linspace(0, self.num_frames - 1,
                                           len(keyframe_prompts)).astype(int)

        frames = []
        keyframe_idx = 0

        for frame_idx in range(self.num_frames):
            # Find surrounding keyframes
            if keyframe_idx < len(keyframe_prompts) - 1:
                next_keyframe_idx = keyframe_indices[keyframe_idx + 1]
            else:
                next_keyframe_idx = self.num_frames

            if frame_idx < next_keyframe_idx or keyframe_idx == len(keyframe_prompts) - 1:
                prompt = keyframe_prompts[keyframe_idx]
            else:
                # Interpolate
                curr_prompt = keyframe_prompts[keyframe_idx]
                next_prompt = keyframe_prompts[keyframe_idx + 1]
                t = (frame_idx - keyframe_indices[keyframe_idx]) / \
                    (next_keyframe_idx - keyframe_indices[keyframe_idx])
                prompt = f"{curr_prompt} transitioning to {next_prompt} ({t:.1f})"

            # Generate frame
            frame = self.generate_frame(prompt, frame_idx)
            frames.append(frame)

            # Update keyframe if needed
            if frame_idx == keyframe_indices[keyframe_idx] and keyframe_idx < len(keyframe_prompts) - 1:
                keyframe_idx += 1

        return frames

    def smooth_frames(self, frames: List[np.ndarray],
                      smooth_window: int = 3) -> List[np.ndarray]:
        """
        Apply temporal smoothing to frames

        Args:
            frames: List of frames
            smooth_window: Size of smoothing window (odd number)

        Returns:
            Smoothed frames
        """
        if smooth_window < 2:
            return frames

        smoothed = []
        half_window = smooth_window // 2

        for i, frame in enumerate(frames):
            # Gather nearby frames
            start_idx = max(0, i - half_window)
            end_idx = min(len(frames), i + half_window + 1)

            # Average
            avg_frame = np.mean(frames[start_idx:end_idx], axis=0)
            smoothed.append(avg_frame)

        return smoothed

    def assemble_video(self, frames: List[np.ndarray], output_path: str,
                       format: str = "mp4") -> str:
        """
        Assemble frames into video file

        Args:
            frames: List of frames
            output_path: Output file path
            format: Video format

        Returns:
            Path to generated video
        """
        try:
            import cv2

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Get video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, self.config.fps,
                                    (width, height))

            for frame in frames:
                # Normalize and convert
                img = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                img = (img * 255).astype(np.uint8)

                # Convert grayscale to BGR for OpenCV
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                writer.write(img)

            writer.release()

            return output_path

        except ImportError:
            # OpenCV not available, save frames as images
            print("OpenCV not available, saving frames as images...")
            output_dir = Path(output_path).parent / "frames"
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(frames):
                import matplotlib.pyplot as plt

                img = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                img = (img * 255).astype(np.uint8)

                plt.imsave(output_dir / f"frame_{i:04d}.png", img, cmap='gray')

            print(f"Frames saved to: {output_dir}")
            return str(output_dir)

    def generate_from_storyboard(self, storyboard: List[dict]) -> List[np.ndarray]:
        """
        Generate video from storyboard

        Args:
            storyboard: List of dicts with 'prompt', 'duration', 'motion'

        Returns:
            List of generated frames
        """
        frames = []

        for scene in storyboard:
            prompt = scene.get('prompt', '')
            duration = scene.get('duration', 1.0)
            motion = scene.get('motion', 'static')

            num_scene_frames = int(duration * self.config.fps)

            for frame_idx in range(num_scene_frames):
                # Add motion to prompt
                motion_prompt = f"{prompt} | motion: {motion}"

                frame = self.generate_frame(motion_prompt, len(frames))
                frames.append(frame)

        # Apply smoothing
        if self.config.smooth_frames > 1:
            frames = self.smooth_frames(frames, self.config.smooth_frames)

        return frames


def run_video_demo():
    """Run video generation demo"""
    print("SACHITRA Video Generation Demo")
    print("=" * 60)

    # Create synthesizer
    config = VideoConfig(fps=24, duration_seconds=3.0)
    synthesizer = VideoSynthesizer(config=config)

    # Define storyboard
    storyboard = [
        {
            'prompt': 'A serene mountain landscape at sunrise',
            'duration': 1.0,
            'motion': 'slow zoom out'
        },
        {
            'prompt': 'A futuristic city with flying vehicles',
            'duration': 1.0,
            'motion': 'pan right'
        },
        {
            'prompt': 'Abstract geometric patterns in motion',
            'duration': 1.0,
            'motion': 'rotation'
        }
    ]

    print(f"\nGenerating {config.duration_seconds}s video at {config.fps} fps")
    print(f"Total frames: {synthesizer.num_frames}")

    # Generate frames
    print("\nGenerating frames...")
    frames = synthesizer.generate_from_storyboard(storyboard)

    print(f"Generated {len(frames)} frames")

    # Save as images (video assembly requires OpenCV)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("\nSaving frames...")
    try:
        import cv2
        video_path = str(output_dir / "demo_video.mp4")
        synthesizer.assemble_video(frames, video_path)
        print(f"Video saved: {video_path}")
    except ImportError:
        print("OpenCV not available, frames saved as images")
        for i, frame in enumerate(frames[:10]):  # Save first 10 frames
            import matplotlib.pyplot as plt
            img = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            plt.imsave(output_dir / f"frame_{i:04d}.png", img, cmap='gray')
        print(f"First 10 frames saved to: {output_dir}")

    print("\nDemo complete!")


if __name__ == "__main__":
    run_video_demo()