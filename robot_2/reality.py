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
from functools import partial

WALL_COLLISION_THRESHOLD = 0.01

@jax.jit
def execute_control(walls: jnp.ndarray, n_sensors: int, settings: "RobotSettings",
                   current_pose: "Pose", control: Tuple[float, float], 
                   key: PRNGKey) -> Tuple["Pose", jnp.ndarray, PRNGKey]:
    """Execute a control command with noise, stopping if we hit a wall"""
    dist, angle = control
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Add noise to motion
    noisy_dist = dist + jax.random.normal(k1) * settings.p_noise
    noisy_angle = angle + jax.random.normal(k2) * settings.hd_noise
    
    # First rotate (can always rotate)
    new_pose = current_pose.rotate(noisy_angle)
    
    # Then try to move forward, checking for collisions
    min_dist = compute_distance_to_wall(walls, new_pose, 0.0, settings.sensor_range)
    
    # Only move as far as we can before hitting a wall
    safe_dist = jnp.minimum(noisy_dist, min_dist - WALL_COLLISION_THRESHOLD)
    safe_dist = jnp.maximum(safe_dist, 0)  # Don't move backwards
    
    new_pose = new_pose.step_along(safe_dist)
    
    # Get sensor readings from new position
    readings, k4 = get_sensor_readings(walls, n_sensors, settings, new_pose, k3)
    
    return new_pose, readings, k4

@jax.jit
def get_sensor_readings(walls: jnp.ndarray, n_sensors: int, settings: "RobotSettings",
                       pose: "Pose", key: PRNGKey) -> Tuple[jnp.ndarray, PRNGKey]:
    """Return noisy distance readings to walls from given pose"""
    MAX_SENSORS = 32  # Fixed maximum
    key, subkey = jax.random.split(key)
    
    # Calculate angles based on n_sensors, but generate MAX_SENSORS of them
    angle_step = 2 * jnp.pi / n_sensors
    angles = jnp.arange(MAX_SENSORS) * angle_step
    noise = jax.random.normal(subkey, (MAX_SENSORS,)) * settings.sensor_noise
    
    readings = jax.vmap(lambda a: compute_distance_to_wall(
        walls, pose, a, settings.sensor_range))(angles)
    
    # Create a mask for the first n_sensors elements
    mask = jnp.arange(MAX_SENSORS) < n_sensors
    
    # Apply mask and pad with zeros
    readings = (readings + noise) * mask
    
    return readings, key

@jax.jit
def compute_distance_to_wall(walls: jnp.ndarray, pose: "Pose", 
                           sensor_angle: float, sensor_range: float) -> float:
    """Compute true distance to nearest wall along sensor ray"""
    if walls.shape[0] == 0:  # No walls
        return sensor_range
        
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
    det = wall_vec[:, 0] * (-ray_dir[1]) - wall_vec[:, 1] * (-ray_dir[0])
    
    # Compute intersection parameters
    u = (to_start[:, 0] * (-ray_dir[1]) - to_start[:, 1] * (-ray_dir[0])) / (det + 1e-10)
    t = (wall_vec[:, 0] * to_start[:, 1] - wall_vec[:, 1] * to_start[:, 0]) / (det + 1e-10)
    
    # Valid intersections: not parallel, in front of ray, within wall segment
    is_valid = (jnp.abs(det) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)
    
    # Find minimum valid distance
    min_dist = jnp.min(jnp.where(is_valid, t, jnp.inf))
    return jnp.where(jnp.isinf(min_dist), sensor_range, min_dist)