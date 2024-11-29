# pyright: reportUnknownMemberType=false
"""
Reality Simulation for Robot Localization Tutorial

This module simulates the "true" state of the world that the robot can only
interact with through noisy sensors and imperfect motion. This separation helps
reinforce that localization is about the robot figuring out where it is using only:

1. What it THINKS it did (control commands)
2. What it can SENSE (noisy sensor readings)
3. What it KNOWS about the world (wall map)

The tutorial code should never peek at _true_pose - it must work only with
the information available through execute_control() and get_sensor_readings().
"""

import jax.numpy as jnp
import jax.lax as lax
from dataclasses import dataclass
from typing import List, Tuple, final, Optional
import jax.random
from jax.random import PRNGKey

@dataclass
class Pose:
    """Represents a robot's position (x,y) and heading (radians)"""
    p: jnp.ndarray  # position [x, y]
    hd: float       # heading in radians

    def step_along(self, s: float) -> "Pose":
        """Move forward by distance s"""
        dp = jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])
        return Pose(self.p + s * dp, self.hd)
    
    def rotate(self, angle: float) -> "Pose":
        """Rotate by angle (in radians)"""
        return Pose(self.p, self.hd + angle)

@final  # Indicates class won't be subclassed
class Reality:
    """
    Simulates the true state of the world, which the robot can only access through
    noisy sensors and imperfect motion.
    
    The robot:
    - Can try to move (but motion will have noise)
    - Can take sensor readings (but they will have noise)
    - Cannot access _true_pose directly
    """

    def __init__(self, walls: jnp.ndarray, motion_noise: float, sensor_noise: float, 
                 initial_pose: Optional[Pose] = None):
        """
        Args:
            walls: JAX array of shape (N, 2, 2) containing wall segments
            motion_noise: Standard deviation of noise added to motion
            sensor_noise: Standard deviation of noise added to sensor readings
            initial_pose: Optional starting pose, defaults to (0.5, 0.5, 0.0)
        """
        self.walls = walls
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise
        self._true_pose = initial_pose if initial_pose is not None else Pose(jnp.array([0.5, 0.5]), 0.0)
        self._key = PRNGKey(0)
    
    def execute_control(self, control: Tuple[float, float]) -> List[float]:
        """Execute a control command with noise, stopping if we hit a wall"""
        dist, angle = control
        # Add noise to motion
        noisy_dist = dist + jax.random.normal(self._key) * self.motion_noise
        noisy_angle = angle + jax.random.normal(self._key) * self.motion_noise
        
        # First rotate (can always rotate)
        self._true_pose = self._true_pose.rotate(noisy_angle)
        
        # Then try to move forward, checking for collisions
        ray_dir = jnp.array([jnp.cos(self._true_pose.hd), jnp.sin(self._true_pose.hd)])
        
        # Use our existing ray-casting to check distance to nearest wall
        min_dist = self._compute_distance_to_wall(0.0)  # 0 angle = forward
        
        # Only move as far as we can before hitting a wall (minus small safety margin)
        safe_dist = jnp.minimum(noisy_dist, min_dist - 0.1)
        safe_dist = jnp.maximum(safe_dist, 0)  # Don't move backwards
        
        self._true_pose = self._true_pose.step_along(safe_dist)
        
        # Return sensor readings from new position
        return self.get_sensor_readings()
    
    def get_sensor_readings(self) -> jnp.ndarray:
        """Return noisy distance readings to walls"""
        angles = jnp.linspace(0, 2*jnp.pi, 8, endpoint=False)  # 8 evenly spaced sensors
        keys = jax.random.split(self._key, 8)
        self._key = keys[0]
        
        def get_reading(key, angle):
            true_dist = self._compute_distance_to_wall(angle)
            return true_dist + jax.random.normal(key) * self.sensor_noise
        
        readings = jax.vmap(get_reading)(keys[1:], angles[:-1])
        return readings
    
    def _compute_distance_to_wall(self, sensor_angle: float) -> jax.Array:
        """Compute true distance to nearest wall along sensor ray using fast 2D ray-segment intersection"""
        ray_start = self._true_pose.p
        ray_angle = self._true_pose.hd + sensor_angle
        ray_dir = jnp.array([jnp.cos(ray_angle), jnp.sin(ray_angle)])
        
        # Vectorized computation for all walls at once
        p1 = self.walls[:, 0]  # Shape: (N, 2)
        p2 = self.walls[:, 1]  # Shape: (N, 2)
        
        # Wall direction vectors
        wall_vec = p2 - p1  # Shape: (N, 2)
        
        # Vector from wall start to ray start
        to_start = ray_start - p1  # Shape: (N, 2)
        
        # Compute determinant (cross product in 2D)
        # This tells us if ray and wall are parallel and their relative orientation
        det = wall_vec[:, 0] * (-ray_dir[1]) - wall_vec[:, 1] * (-ray_dir[0])
        
        # Compute intersection parameters
        u = (to_start[:, 0] * (-ray_dir[1]) - to_start[:, 1] * (-ray_dir[0])) / (det + 1e-10)
        t = (wall_vec[:, 0] * to_start[:, 1] - wall_vec[:, 1] * to_start[:, 0]) / (det + 1e-10)
        
        # Valid intersections: not parallel, in front of ray, within wall segment
        is_valid = (jnp.abs(det) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)
        
        # Find minimum valid distance
        min_dist = jnp.min(jnp.where(is_valid, t, jnp.inf))
        return jnp.where(jnp.isinf(min_dist), 10.0, min_dist)