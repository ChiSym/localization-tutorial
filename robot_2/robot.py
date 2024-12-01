import jax
import jax.numpy as jnp
from functools import partial
from penzai import pz
import genjax
from typing import List, Tuple
from robot_2.reality import execute_control
from jax.random import PRNGKey
from dataclasses import dataclass


@pz.pytree_dataclass
class Pose(genjax.PythonicPytree):
    """Robot pose with position and heading"""
    p: jax.Array  # [x, y]
    hd: float     # heading in radians
    
    def dp(self):
        """Get direction vector from heading"""
        return jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])
        
    def step_along(self, s: float) -> "Pose":
        """Move forward by distance s"""
        return Pose(self.p + s * self.dp(), self.hd)
    
    def rotate(self, angle: float) -> "Pose":
        """Rotate by angle (in radians)"""
        return Pose(self.p, self.hd + angle)
    
@pz.pytree_dataclass
class RobotSettings(genjax.PythonicPytree):
    """Robot configuration and uncertainty settings"""
    p_noise: float = 0.1        # Position noise
    hd_noise: float = 0.1       # Heading noise
    sensor_noise: float = 0.1   # Sensor noise
    sensor_range: float = 10.0  # Maximum sensor range

def path_to_controls(path_points: List[List[float]]) -> jnp.ndarray:
    """Convert a series of points into (distance, angle) control pairs"""
    points = jnp.array([p[:2] for p in path_points])
    deltas = points[1:] - points[:-1]
    distances = jnp.linalg.norm(deltas, axis=1)
    angles = jnp.arctan2(deltas[:, 1], deltas[:, 0])
    angle_changes = jnp.diff(angles, prepend=0.0)
    return jnp.stack([distances, angle_changes], axis=1)

def sample_single_path(carry, control, walls, n_sensors, settings):
    """Single step of path sampling that can be used with scan"""
    pose, key = carry
    pose, _, key = execute_control(
        walls=walls,
        n_sensors=n_sensors,
        settings=settings,
        current_pose=pose,
        control=control,
        key=key
    )
    return (pose, key), pose.p

@partial(jax.jit, static_argnums=(1))
def sample_possible_paths(key: jnp.ndarray, n_paths: int, n_sensors: int,
                         robot_path: jnp.ndarray, walls: jnp.ndarray, 
                         settings: RobotSettings):
    """Generate n possible paths given the planned path, respecting walls"""
    path_points = robot_path[:, :2]
    controls = path_to_controls(path_points)
    
    start_point = path_points[0]
    start_pose = Pose(jnp.array(start_point, dtype=jnp.float32), 0.0)
    
    keys = jax.random.split(key, n_paths)
    
    def sample_path_scan(key):
        init_carry = (start_pose, key)
        (final_pose, final_key), path_points = jax.lax.scan(
            lambda carry, control: sample_single_path(carry, control, walls, n_sensors, settings),
            init_carry,
            controls
        )
        return jnp.concatenate([start_pose.p[None, :], path_points], axis=0)
    
    paths = jax.vmap(sample_path_scan)(keys)
    return paths

@jax.jit
def simulate_robot_path(start_pose: Pose, n_sensors: int, controls: jnp.ndarray, 
                       walls: jnp.ndarray, settings: RobotSettings, key: jnp.ndarray):
    """Simulate robot path with noise and sensor readings"""
    def step_fn(carry, control):
        pose, k = carry
        new_pose, readings, new_key = execute_control(
            walls=walls,
            n_sensors=n_sensors, 
            settings=settings,
            current_pose=pose,
            control=control,
            key=k
        )
        return (new_pose, new_key), (new_pose, readings)
    
    return jax.lax.scan(step_fn, (start_pose, key), controls)
