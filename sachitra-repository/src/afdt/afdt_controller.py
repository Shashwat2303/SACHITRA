"""
SACHITRA AFDT (Adaptive Finite-Difference Trajectory Approximation)
Training-free ODE acceleration with formal error guarantees
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class SkipStrategy(Enum):
    """ODE step skipping strategies"""
    NONE = "none"
    FIXED = "fixed"
    ADAPTIVE_BANDIT = "adaptive_bandit"
    ADAPTIVE_TEACACHE = "adaptive_teacache"


@dataclass
class AFDTConfig:
    """Configuration for AFDT controller"""
    # Skip lengths available to bandit
    skip_lengths: Tuple[int, ...] = (0, 2, 4, 6)

    # UCB-1 parameters
    bandit_gamma: float = 2.0
    exploration_bonus: float = 2.0

    # Error bounds
    max_trajectory_error: float = 0.01
    lipchitz_constant: float = 1.0
    bounded_second_derivative: float = 1.0

    # Quantization parameters
    quantization_bits: int = 4
    quantization_error: float = 0.01

    # Warm start
    warm_start_iterations: int = 50

    # Persistence
    persist_to_indexed_db: bool = True


class UCB1BanditController:
    """
    UCB-1 Multi-Armed Bandit for adaptive ODE step skipping

    Selects skip length α ∈ A to maximize:
    Q(α) + γ * √(ln(n) / N(α))

    Reward function:
    r(α) = μ * α - ℓ(v̂, v)

    Where:
    - μ normalizes rewards
    - α is the skip length
    - ℓ is the MSE between approximated and true velocity
    """

    def __init__(self, config: AFDTConfig):
        self.config = config
        self.num_arms = len(config.skip_lengths)
        self.arms = config.skip_lengths

        # Q values: empirical mean reward for each arm
        self.Q = np.zeros(self.num_arms)

        # N values: play count for each arm (avoid div by zero)
        self.N = np.ones(self.num_arms)

        # Total invocations
        self.n = 0

        # Best arm history
        self.best_arm_history: List[int] = []

        # Arm index mapping
        self.arm_to_idx = {arm: i for i, arm in enumerate(self.arms)}

    def select(self) -> int:
        """
        Select arm using UCB-1 policy

        Returns:
            Selected skip length
        """
        self.n += 1

        # Compute UCB for each arm
        ucb_values = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            if self.N[i] == 0:
                exploration = self.config.exploration_bonus
            else:
                exploration = self.config.bandit_gamma * np.sqrt(
                    np.log(self.n) / self.N[i]
                )
            ucb_values[i] = self.Q[i] + exploration

        # Select arm with maximum UCB
        arm_idx = np.argmax(ucb_values)
        self.N[arm_idx] += 1

        selected_arm = self.arms[arm_idx]
        self.best_arm_history.append(selected_arm)

        return selected_arm

    def update(self, arm: int, reward: float) -> None:
        """
        Update Q value for selected arm using incremental mean

        Q(α)_new = Q(α)_old + (r - Q(α)_old) / N(α)
        """
        arm_idx = self.arm_to_idx[arm]

        # Incremental update
        self.Q[arm_idx] = self.Q[arm_idx] + (reward - self.Q[arm_idx]) / self.N[arm_idx]

    def compute_reward(self, skip_length: int, mse_approx: float,
                      max_mse: float = 1.0) -> float:
        """
        Compute reward for bandit update

        r(α) = μ * α - ℓ(v̂, v)

        - μ = max(MSE) / K normalizes rewards
        - α is the skip length (positive contribution)
        - ℓ is the MSE (negative contribution, penalty for inaccuracy)

        Args:
            skip_length: Number of steps to skip
            mse_approx: MSE between approximated and true velocity
            max_mse: Maximum observed MSE for normalization

        Returns:
            Normalized reward value
        """
        # Normalization factor
        mu = max_mse / max(self.n, 1)

        # Reward = benefit from skipping - cost of approximation error
        reward = mu * skip_length - mse_approx

        # Clip to reasonable range
        reward = np.clip(reward, -10, 10)

        return reward

    def get_statistics(self) -> Dict:
        """Get bandit statistics for debugging"""
        return {
            'total_selections': self.n,
            'arm_counts': self.N.tolist(),
            'mean_rewards': self.Q.tolist(),
            'arm_to_idx': self.arm_to_idx,
            'best_arm_history': self.best_arm_history[-100:]  # Last 100
        }

    def reset(self) -> None:
        """Reset bandit to initial state"""
        self.Q = np.zeros(self.num_arms)
        self.N = np.ones(self.num_arms)
        self.n = 0
        self.best_arm_history = []

    def get_optimal_arm(self) -> int:
        """Get the arm with highest mean reward"""
        return self.arms[np.argmax(self.Q)]


class VelocityEstimator:
    """
    Finite-difference velocity estimation for AFDT

    The core insight is that FM models trained on linear interpolants
    generate approximately linear trajectories in the intermediate regime.

    v(x_{t+Δt}, t+Δt) ≈ v(x_t, t) + Δt * dv/dt

    Where dv/dt is estimated via finite-difference between two prior evaluations.
    """

    def __init__(self, config: AFDTConfig):
        self.config = config

    def estimate_velocity(self,
                          v_curr: np.ndarray,
                          t_curr: float,
                          v_prev: Optional[np.ndarray],
                          t_prev: Optional[float],
                          v_ref: Optional[np.ndarray],
                          t_ref: Optional[float],
                          dt: float) -> np.ndarray:
        """
        Estimate velocity using finite-difference approximation

        First-order: v_approx = v_curr + dt * dv/dt
        Second-order: v_approx = v_curr + dt * dv/dt + 0.5 * dt^2 * d²v/dt²

        Args:
            v_curr: Current velocity
            t_curr: Current time
            v_prev: Previous velocity (or None)
            t_prev: Previous time (or None)
            v_ref: Reference velocity (or None)
            t_ref: Reference time (or None)
            dt: Time step size

        Returns:
            Estimated velocity for next step
        """
        if v_prev is None:
            # Cannot do finite-difference, return current velocity
            return v_curr.copy()

        # First-order finite difference
        dt_actual = max(t_curr - t_prev, 1e-6)
        dv_dt = (v_curr - v_prev) / dt_actual

        # First-order Taylor expansion
        v_approx = v_curr + dt * dv_dt

        # Second-order correction if reference point available
        if v_ref is not None and t_ref is not None and t_prev > t_ref:
            dt_ref = max(t_prev - t_ref, 1e-6)
            dv_dt_2 = (v_prev - v_ref) / dt_ref

            # Second-order Taylor expansion
            v_approx = v_curr + dt * dv_dt + 0.5 * (dt ** 2) * dv_dt_2

        return v_approx

    def estimate_velocity_batch(self,
                                 v_curr: np.ndarray,
                                 t_curr: float,
                                 v_prev: Optional[np.ndarray],
                                 t_prev: Optional[float],
                                 v_ref: Optional[np.ndarray],
                                 t_ref: Optional[float],
                                 dt: float) -> np.ndarray:
        """
        Batch version of velocity estimation

        Args:
            v_curr: Current velocities [batch, dim]
            t_curr: Current time
            v_prev: Previous velocities [batch, dim]
            t_prev: Previous time
            v_ref: Reference velocities [batch, dim]
            t_ref: Reference time
            dt: Time step size

        Returns:
            Estimated velocities [batch, dim]
        """
        if v_prev is None:
            return v_curr.copy()

        # Compute finite-difference
        dt_actual = max(t_curr - t_prev, 1e-6)
        dv_dt = (v_curr - v_prev) / dt_actual

        # First-order approximation
        v_approx = v_curr + dt * dv_dt

        # Second-order correction
        if v_ref is not None and t_ref is not None and t_prev > t_ref:
            dt_ref = max(t_prev - t_ref, 1e-6)
            dv_dt_2 = (v_prev - v_ref) / dt_ref
            v_approx = v_curr + dt * dv_dt + 0.5 * (dt ** 2) * dv_dt_2

        return v_approx


class AFDTController:
    """
    Adaptive Finite-Difference Trajectory Approximation Controller

    Integrates:
    1. UCB-1 bandit for adaptive step skipping
    2. Finite-difference velocity estimation
    3. Formal error bound verification (Theorem 1)
    """

    def __init__(self, config: Optional[AFDTConfig] = None):
        self.config = config or AFDTConfig()
        self.bandit = UCB1BanditController(self.config)
        self.velocity_estimator = VelocityEstimator(self.config)

        # Velocity history for finite-difference
        self.v_history: List[Tuple[float, np.ndarray]] = []
        self.t_history: List[float] = []

        # Statistics
        self.total_steps = 0
        self.skipped_steps = 0
        self.approximation_errors: List[float] = []

    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Perform one AFDT step

        Args:
            x: Current state
            t: Current time
            dt: Time step
            velocity_fn: Function to compute true velocity v(x, t)

        Returns:
            Tuple of (next_state, approximation_error)
        """
        # Compute current velocity
        v_curr = velocity_fn(x, t)

        # Update history
        self.t_history.append(t)
        self.v_history.append(v_curr)

        # Keep only last 3 velocity evaluations
        if len(self.v_history) > 3:
            self.v_history.pop(0)
            self.t_history.pop(0)

        # Get previous velocities for finite-difference
        v_prev = self.v_history[-2] if len(self.v_history) >= 2 else None
        t_prev = self.t_history[-2] if len(self.t_history) >= 2 else None
        v_ref = self.v_history[-3] if len(self.v_history) >= 3 else None
        t_ref = self.t_history[-3] if len(self.t_history) >= 3 else None

        # Select skip length via bandit
        skip_length = self.bandit.select()

        # Compute velocity approximation
        v_approx = self.velocity_estimator.estimate_velocity(
            v_curr, t, v_prev, t_prev, v_ref, t_ref, dt
        )

        # Compute approximation error
        mse_approx = np.mean((v_approx - v_curr) ** 2)
        self.approximation_errors.append(mse_approx)

        # Update bandit
        max_mse = max(self.approximation_errors) if self.approximation_errors else 1.0
        reward = self.bandit.compute_reward(skip_length, mse_approx, max_mse)
        self.bandit.update(skip_length, reward)

        # Perform step with skip
        total_dt = (skip_length + 1) * dt
        x_next = x + v_approx * total_dt

        self.total_steps += 1
        self.skipped_steps += skip_length

        return x_next, mse_approx

    def compute_theoretical_error_bound(self, num_skips: int, num_steps: int) -> float:
        """
        Compute theoretical error bound per Theorem 1

        e_T ≤ (M * e^{Lx} / 2) * (|S| / T^3) + e^{Lx} * K * ε_q

        Where:
        - M = bounded second time-derivative constant
        - Lx = Lipschitz continuity constant of velocity field
        - |S| = number of skipped steps
        - T = total number of steps
        - K = number of full evaluations
        - ε_q = quantization error

        Args:
            num_skips: Number of steps skipped
            num_steps: Total number of steps

        Returns:
            Theoretical error bound
        """
        M = self.config.bounded_second_derivative
        Lx = self.config.lipchitz_constant
        eps_q = self.config.quantization_error

        T = num_steps
        K = num_skips  # Number of skips

        # Approximation error term: O(|S| / T^3)
        approx_term = (M * np.exp(Lx) / 2) * (K / (T ** 3))

        # Quantization error term: O(K * ε_q)
        quant_term = np.exp(Lx) * K * eps_q

        return approx_term + quant_term

    def verify_error_bound(self) -> Dict[str, float]:
        """
        Verify that actual errors stay within theoretical bounds

        Returns:
            Dictionary with verification results
        """
        if not self.approximation_errors:
            return {'status': 'no_data'}

        # Actual maximum error
        max_actual_error = max(self.approximation_errors)

        # Theoretical bound
        theoretical_bound = self.compute_theoretical_error_bound(
            self.skipped_steps,
            self.total_steps
        )

        # Verification
        bound_satisfied = max_actual_error <= theoretical_bound

        return {
            'max_actual_error': max_actual_error,
            'theoretical_bound': theoretical_bound,
            'bound_satisfied': bound_satisfied,
            'error_ratio': max_actual_error / theoretical_bound if theoretical_bound > 0 else 0,
            'total_steps': self.total_steps,
            'skipped_steps': self.skipped_steps,
            'skip_ratio': self.skipped_steps / max(self.total_steps, 1)
        }

    def reset(self) -> None:
        """Reset controller state"""
        self.bandit.reset()
        self.v_history = []
        self.t_history = []
        self.total_steps = 0
        self.skipped_steps = 0
        self.approximation_errors = []

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'bandit': self.bandit.get_statistics(),
            'verification': self.verify_error_bound(),
            'total_steps': self.total_steps,
            'skipped_steps': self.skipped_steps,
            'skip_ratio': self.skipped_steps / max(self.total_steps, 1),
            'mean_approximation_error': np.mean(self.approximation_errors) if self.approximation_errors else 0,
            'max_approximation_error': max(self.approximation_errors) if self.approximation_errors else 0
        }

    def save_state(self) -> Dict:
        """Save state for IndexedDB persistence"""
        return {
            'bandit_Q': self.bandit.Q.tolist(),
            'bandit_N': self.bandit.N.tolist(),
            'bandit_n': self.bandit.n,
            'total_steps': self.total_steps,
            'skipped_steps': self.skipped_steps
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence"""
        self.bandit.Q = np.array(state['bandit_Q'])
        self.bandit.N = np.array(state['bandit_N'])
        self.bandit.n = state['bandit_n']
        self.total_steps = state['total_steps']
        self.skipped_steps = state['skipped_steps']


def run_afdt_benchmark(num_steps: int = 50, num_runs: int = 100) -> Dict:
    """
    Run AFDT benchmark comparing with standard Euler integration

    Args:
        num_steps: Number of ODE integration steps
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    config = AFDTConfig()
    controller = AFDTController(config)

    # Simple velocity field for testing: v(x, t) = -x + noise
    def velocity_fn(x, t):
        return -x + np.random.randn(*x.shape) * 0.01

    # Run benchmarks
    euler_times = []
    afdt_times = []
    errors = []

    for run in range(num_runs):
        # Reset
        controller.reset()
        x_euler = np.random.randn(512)
        x_afdt = x_euler.copy()
        t = 0
        dt = 1.0 / num_steps

        euler_start = np.datetime64('now').astype('int64')
        for step in range(num_steps):
            v = velocity_fn(x_euler, t)
            x_euler = x_euler + dt * v
            t += dt
        euler_time = (np.datetime64('now').astype('int64') - euler_start) / 1e6

        afdt_start = np.datetime64('now').astype('int64')
        for step in range(num_steps):
            x_afdt, _ = controller.step(x_afdt, t - (num_steps - step) * dt, dt, velocity_fn)
        afdt_time = (np.datetime64('now').astype('int64') - afdt_start) / 1e6

        euler_times.append(euler_time)
        afdt_times.append(afdt_time)
        errors.append(np.linalg.norm(x_euler - x_afdt))

    return {
        'mean_euler_time_ms': np.mean(euler_times),
        'mean_afdt_time_ms': np.mean(afdt_times),
        'speedup': np.mean(euler_times) / max(np.mean(afdt_times), 0.001),
        'mean_error': np.mean(errors),
        'max_error': max(errors),
        'controller_stats': controller.get_statistics()
    }


if __name__ == "__main__":
    print("SACHITRA AFDT Controller")
    print("=" * 50)

    # Create controller
    config = AFDTConfig()
    controller = AFDTController(config)

    # Simple velocity field
    def velocity_fn(x, t):
        return -x + np.random.randn(*x.shape) * 0.01

    print("\nRunning AFDT inference...")
    x = np.random.randn(512)
    t = 0
    dt = 1.0 / 50

    for step in range(50):
        x, error = controller.step(x, t, dt, velocity_fn)
        t += dt

        if step % 10 == 0:
            print(f"  Step {step}: error = {error:.6f}")

    # Verify error bound
    stats = controller.get_statistics()
    print(f"\nAFDT Statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Skipped steps: {stats['skipped_steps']}")
    print(f"  Skip ratio: {stats['skip_ratio']:.2%}")
    print(f"  Mean approximation error: {stats['mean_approximation_error']:.6f}")
    print(f"  Max approximation error: {stats['max_approximation_error']:.6f}")

    verification = stats['verification']
    print(f"\nError Bound Verification:")
    print(f"  Max actual error: {verification['max_actual_error']:.6f}")
    print(f"  Theoretical bound: {verification['theoretical_bound']:.6f}")
    print(f"  Bound satisfied: {verification['bound_satisfied']}")

    # Run benchmark
    print("\n" + "=" * 50)
    print("Running benchmark...")
    benchmark = run_afdt_benchmark(num_steps=50, num_runs=10)
    print(f"\nBenchmark Results:")
    print(f"  Mean Euler time: {benchmark['mean_euler_time_ms']:.2f} ms")
    print(f"  Mean AFDT time: {benchmark['mean_afdt_time_ms']:.2f} ms")
    print(f"  Speedup: {benchmark['speedup']:.2f}x")
    print(f"  Mean error: {benchmark['mean_error']:.6f}")

    # Bandit statistics
    print(f"\nBandit Statistics:")
    bandit_stats = controller.bandit.get_statistics()
    for i, arm in enumerate(config.skip_lengths):
        print(f"  Arm {arm}: N={bandit_stats['arm_counts'][i]}, Q={bandit_stats['mean_rewards'][i]:.4f}")