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

    def __init__(self, walls: List[Tuple[float, float]], motion_noise: float, sensor_noise: float):
        self.walls = walls
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise
        self._true_pose = Pose(jnp.array([0.5, 0.5]), 0.0)
        self._key = PRNGKey(0)
    
    def execute_control(self, control: Tuple[float, float]):
        """
        Execute a control command with noise
        control: (forward_dist, rotation_angle)
        """
        dist, angle = control
        # Add noise to motion
        self._key, k1, k2 = jax.random.split(self._key, 3)
        noisy_dist = dist + jax.random.normal(k1) * self.motion_noise
        noisy_angle = angle + jax.random.normal(k2) * self.motion_noise
        
        # Update true pose
        self._true_pose = self._true_pose.step_along(float(noisy_dist)).rotate(float(noisy_angle))
        
        # Return only sensor readings
        return self.get_sensor_readings()
    
    def get_sensor_readings(self) -> List[float]:
        """Return noisy distance readings to walls"""
        angles = jnp.linspace(0, 2*jnp.pi, 8)  # 8 sensors around robot
        keys = jax.random.split(self._key, 8)
        self._key = keys[0]
        
        def get_reading(key, angle):
            true_dist = self._compute_distance_to_wall(angle)
            return true_dist + jax.random.normal(key) * self.sensor_noise
        
        readings = jax.vmap(get_reading)(keys[1:], angles)
        return readings.tolist()
    
    def _compute_distance_to_wall(self, sensor_angle: float) -> float:
        """Compute true distance to nearest wall along sensor ray"""
        # Ray starts at robot position
        ray_start = self._true_pose.p
        # Ray direction based on robot heading plus sensor angle
        ray_angle = self._true_pose.hd + sensor_angle
        ray_dir = jnp.array([jnp.cos(ray_angle), jnp.sin(ray_angle)])
        
        def check_wall_intersection(min_dist, i):
            p1 = jnp.array(self.walls[i])
            p2 = jnp.array(self.walls[i+1])
            
            # Wall vector
            wall = p2 - p1
            # Vector from wall start to ray start
            s = ray_start - p1
            
            # Compute intersection using parametric equations
            # Ray: ray_start + t*ray_dir
            # Wall: p1 + u*wall
            wall_norm = wall/jnp.linalg.norm(wall)
            denom = jnp.cross(ray_dir, wall_norm)
            
            t = jnp.cross(wall_norm, s) / (denom + 1e-10)
            u = jnp.cross(ray_dir, s) / (denom + 1e-10)
            
            # Check if intersection is valid (in front of ray and within wall segment)
            is_valid = (jnp.abs(denom) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)
            return jnp.where(is_valid, jnp.minimum(min_dist, t), min_dist)
        
        min_dist = lax.fori_loop(0, len(self.walls)-1, check_wall_intersection, jnp.inf)
        return float(jnp.where(jnp.isinf(min_dist), 100.0, min_dist).item())