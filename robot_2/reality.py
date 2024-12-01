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

def execute_control(walls: jnp.ndarray, motion_noise: float, sensor_noise: float, 
                   current_pose: Pose, control: Tuple[float, float], 
                   key: PRNGKey) -> Tuple[Pose, jnp.ndarray, PRNGKey]:
    """Execute a control command with noise, stopping if we hit a wall
    
    Args:
        walls: JAX array of shape (N, 2, 2) containing wall segments
        motion_noise: Standard deviation of noise added to motion
        sensor_noise: Standard deviation of noise added to sensor readings
        current_pose: The pose to start from
        control: (distance, angle) tuple of motion command
        key: JAX random key for noise generation
        
    Returns:
        (new_pose, sensor_readings, new_key) tuple
    """
    dist, angle = control
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Add noise to motion
    noisy_dist = dist + jax.random.normal(k1) * motion_noise
    noisy_angle = angle + jax.random.normal(k2) * motion_noise
    
    # First rotate (can always rotate)
    new_pose = current_pose.rotate(noisy_angle)
    
    # Then try to move forward, checking for collisions
    min_dist = compute_distance_to_wall(walls, new_pose, 0.0)  # 0 angle = forward
    
    # Only move as far as we can before hitting a wall
    safe_dist = jnp.minimum(noisy_dist, min_dist - 0.1)
    safe_dist = jnp.maximum(safe_dist, 0)  # Don't move backwards
    
    new_pose = new_pose.step_along(safe_dist)
    
    # Get sensor readings from new position
    readings, k4 = get_sensor_readings(walls, sensor_noise, new_pose, k3)
    
    return new_pose, readings, k4

def get_sensor_readings(walls: jnp.ndarray, sensor_noise: float, 
                       pose: Pose, key: PRNGKey) -> Tuple[jnp.ndarray, PRNGKey]:
    """Return noisy distance readings to walls from given pose"""
    angles = jnp.linspace(0, 2*jnp.pi, 8, endpoint=False)
    keys = jax.random.split(key, 8)
    
    def get_reading(key, angle):
        true_dist = compute_distance_to_wall(walls, pose, angle)
        return true_dist + jax.random.normal(key) * sensor_noise
    
    readings = jax.vmap(get_reading)(keys[1:], angles[:-1])
    return readings, keys[0]

def compute_distance_to_wall(walls: jnp.ndarray, pose: Pose, sensor_angle: float) -> float:
    """Compute true distance to nearest wall along sensor ray"""
    if walls.shape[0] == 0:  # No walls
        return 10.0  # Return max sensor range
        
    ray_start = pose.p
    ray_dir = jnp.array([
        jnp.cos(pose.hd + sensor_angle),
        jnp.sin(pose.hd + sensor_angle)
    ])
    
    # Vectorized computation for all walls at once
    p1 = walls[:, 0]  # Shape: (N, 2)
    p2 = walls[:, 1]  # Shape: (N, 2)
    
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