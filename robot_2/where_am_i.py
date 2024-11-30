# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Robot Localization: A Robot's Perspective
#
# Imagine you're a robot. You have:
# 1. A map of walls in your environment
# 2. A plan of where to go ("move forward 1m, turn right 30°")
# 3. Distance sensors that measure how far walls are
#
# **Your Challenge**: Figure out where you actually are!
#
# ## Why is this Hard?
#
# You can't just follow your plan perfectly because:
# - Wheels slip and drift
# - Sensors give noisy readings
# - Small errors add up over time
#
# ## Try It Yourself
#
# 1. First, create the environment:
#    - Draw some walls by clicking
#
# 2. Then, plan the robot's path:
#    - Draw where you want the robot to go
#    - This becomes a series of movement commands
#
# 3. Watch what happens:
#    - Blue line: What the robot THINKS it's doing (following commands perfectly)
#    - Red rays: What the robot actually SEES (sensor readings)
#    - Blue cloud: Where the robot MIGHT be (uncertainty)
#    - Green line: Where the robot figures it ACTUALLY is

# %%
# pyright: reportUnusedExpression=false
# pyright: reportUnknownMemberType=false

import genstudio.plot as Plot
from genstudio.plot import js
import numpy as np
import jax.numpy as jnp
from typing import TypedDict, List, Tuple, Any 

import robot_2.reality as reality
import jax
import jax.numpy as jnp
import genjax
from genjax import normal, mv_normal_diag
from penzai import pz  # Import penzai for pytree dataclasses
from typing import List, Tuple

@pz.pytree_dataclass
class MotionSettings(genjax.PythonicPytree):
    """Settings for motion uncertainty"""
    p_noise: float = 0.1  # Position noise
    hd_noise: float = 0.1  # Heading noise

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

@genjax.gen
def step_proposal(motion_settings: MotionSettings, start_pose: Pose, control: Tuple[float, float], walls: jnp.ndarray) -> Pose:
    """Generate a noisy next pose given current pose and control, respecting walls"""
    dist, angle = control
    
    # Sample noisy heading first
    hd = normal(
        start_pose.hd + angle, 
        motion_settings.hd_noise
    ) @ "hd"
    
    # Calculate noisy forward distance
    noisy_dist = normal(
        dist,
        motion_settings.p_noise
    ) @ "dist"
    
    # Simplified collision check using just the wall segments
    ray_start = start_pose.p
    ray_dir = jnp.array([jnp.cos(hd), jnp.sin(hd)])
    
    # Vectorized ray-wall intersection (copied from Reality class)
    p1 = walls[:, 0]  # Shape: (N, 2)
    p2 = walls[:, 1]  # Shape: (N, 2)
    wall_vec = p2 - p1
    to_start = ray_start - p1
    
    det = wall_vec[:, 0] * (-ray_dir[1]) - wall_vec[:, 1] * (-ray_dir[0])
    u = (to_start[:, 0] * (-ray_dir[1]) - to_start[:, 1] * (-ray_dir[0])) / (det + 1e-10)
    t = (wall_vec[:, 0] * to_start[:, 1] - wall_vec[:, 1] * to_start[:, 0]) / (det + 1e-10)
    
    is_valid = (jnp.abs(det) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)
    min_dist = jnp.min(jnp.where(is_valid, t, jnp.inf))
    min_dist = jnp.where(jnp.isinf(min_dist), 10.0, min_dist)
    
    # Limit distance to avoid wall collision
    safe_dist = jnp.minimum(noisy_dist, min_dist - 0.1)
    safe_dist = jnp.maximum(safe_dist, 0)  # Don't move backwards
    
    # Calculate final position
    p = start_pose.p + safe_dist * ray_dir
    
    return Pose(p, hd)

@genjax.gen
def generate_path(motion_settings: MotionSettings, start_pose: Pose, controls: List[Tuple[float, float]], walls: jnp.ndarray):
    """Generate a complete path by sampling noisy steps, respecting walls"""
    pose = start_pose
    path = [pose]
    
    for control in controls:
        pose = step_proposal(motion_settings, pose, control, walls) @ f"step_{len(path)}"
        path.append(pose)
    
    return path

def sample_possible_paths(key: jax.random.PRNGKey, n_paths: int, robot_path: List[List[float]], walls: List[List[float]]):
    """Generate n possible paths given the planned path, respecting walls"""
    controls = path_to_controls(robot_path)
    start_point = robot_path[0]
    start_pose = Pose(jnp.array(start_point[:2]), 0.0)
    settings = MotionSettings()
    
    # Convert walls to JAX array format
    wall_segments = convert_walls_to_jax(walls)
    
    # Split key for multiple samples
    keys = jax.random.split(key, n_paths)
    
    # Sample paths using GenJAX
    def sample_single_path(k):
        trace = generate_path.simulate(k, (settings, start_pose, controls, wall_segments))
        poses = trace.get_retval()
        return jnp.stack([pose.p for pose in poses])
    
    paths = jax.vmap(sample_single_path)(keys)
    return paths

_gensym_counter = 0

WALL_WIDTH=6
PATH_WIDTH=6

def gensym(prefix: str = "g") -> str:
    """Generate a unique symbol with an optional prefix, similar to Clojure's gensym."""
    global _gensym_counter
    _gensym_counter += 1
    return f"{prefix}{_gensym_counter}"

def drawing_system(on_complete):
    key = gensym("current_line")
    line =  Plot.line(
                js(f"$state.{key}"),
                stroke="#ccc",
                strokeWidth=4,
                strokeDasharray="4")
    
    events = Plot.events({
        "_initialState": Plot.initialState({key: []}),
        "onDrawStart": js(f"""(e) => {{
            $state.{key} = [[e.x, e.y, e.startTime]];
        }}"""),
        "onDraw": js(f"""(e) => {{
            if ($state.{key}.length > 0) {{
                const last = $state.{key}[$state.{key}.length - 1];
                const dx = e.x - last[0];
                const dy = e.y - last[1];
                // Only add point if moved more than threshold distance
                if (Math.sqrt(dx*dx + dy*dy) > 0.2) {{
                    $state.update(['{key}', 'append', [e.x, e.y, e.startTime]]);
                }}
            }}
        }}"""),
        "onDrawEnd": js(f"""(e) => {{
            if ($state.{key}.length > 1) {{
                // Simplify line by keeping only every 3rd point
                // keep this, we may re-enable later
                //const simplified = $state.{key}.filter((_, i) => i % 3 === 0);
                %1($state.{key})
            }}
            $state.{key} = [];
        }}""", on_complete)
    })
    return line + events 

sliders = (
     Plot.Slider(
            "sensor_noise",
            range=[0, 1],
            step=0.05,
            label="Sensor Noise:", 
            showValue=True
        )
     & Plot.Slider(
            "motion_noise",
            range=[0, 1],
            step=0.05,
            label="Motion Noise:",
            showValue=True
        )
)

def initial_walls():
    return [
            # Frame around domain (timestamp 0)
            [0, 0, 0], [10, 0, 0],  # Bottom
            [10, 0, 0], [10, 10, 0],  # Right
            [10, 10, 0], [0, 10, 0],  # Top
            [0, 10, 0], [0, 0, 0],  # Left
        ]

initial_state = {
        "walls": initial_walls(),
        "robot_pose": {"x": 0.5, "y": 0.5, "heading": 0},
        "sensor_noise": 0.1,
        "motion_noise": 0.1,
        "show_sensors": True,
        "selected_tool": "path",
        "robot_path": [],
        "estimated_pose": None,
        "sensor_readings": [],
        "show_uncertainty": True,
        "show_true_position": True  # New flag for toggling true position visibility
    }

sensor_rays = Plot.line(
                js("""
                   Array.from($state.sensor_readings).flatMap((r, i) => {
                    const heading = $state.robot_pose.heading || 0;
                    const angle = heading + (i * Math.PI * 2) / 8;
                    const x = $state.robot_pose.x;
                    const y = $state.robot_pose.y;
                    return [
                        [x, y, i],
                        [x + r * Math.cos(angle), 
                         y + r * Math.sin(angle), i]
                    ]
                })
                   """),
                z="2",
                stroke="red",
                strokeWidth=1
            )

true_path = Plot.line(
                js("$state.true_path"),
                stroke=Plot.constantly("True Path"),
                strokeWidth=2
            )

planned_path = Plot.line(
                    js("$state.robot_path"),
                    stroke=Plot.constantly("Robot Path"),
                    strokeWidth=PATH_WIDTH),

canvas = (
            # Draw completed walls
            Plot.line(
                js("$state.walls"),
                stroke=Plot.constantly("Walls"),
                strokeWidth=WALL_WIDTH,
                z="2", 
                render=Plot.renderChildEvents({"onClick": js("""(e) => {
                    const zs = new Set($state.walls.map(w => w[2]));
                    const targetZ = [...zs][e.index];
                    $state.walls = $state.walls.filter(([x, y, z]) => z !== targetZ)
                    }""")})
            )
            # Draw current line being drawn
            + drawing_system(Plot.js("""(line) => {
                if ($state.selected_tool === 'walls') {
                    $state.update(['walls', 'concat', line]);
                } else if ($state.selected_tool === 'path') {
                    $state.update(['robot_path', 'reset', line]);
                }
                }"""))
            + planned_path
            
            # Draw robot
            + Plot.cond(
                js("$state.show_true_position"), 
                [Plot.text(
                    js("[[$state.robot_pose.x, $state.robot_pose.y]]"),
                    text=Plot.constantly("🤖"),
                    fontSize=30,
                    textAnchor="middle",
                    dy="-0.35em",
                    rotate=js("$state.robot_pose.heading * 180 / Math.PI")), 
                    true_path,
                sensor_rays
                ]
                )
            + Plot.domain([0, 10], [0, 10])
            + Plot.grid()
            + Plot.aspectRatio(1)
            + Plot.colorMap({
                "Walls": "#666",
                "Sensor Rays": "red",
                "True Path": "green",
                "Robot Path": "blue",
            })
            + Plot.colorLegend()
            + Plot.line(
        js("""
           if (!$state.show_debug || !$state.possible_paths) {return [];};
           return $state.possible_paths.flatMap((path, pathIdx) => 
               path.map(([x, y]) => [x, y, pathIdx])
           )
        """, expression=False),
        stroke="blue",
        strokeOpacity=0.2
    )
            + Plot.clip()
        )


def convert_walls_to_jax(walls_list: List[List[float]]) -> jnp.ndarray:
    """Convert wall vertices from UI format to JAX array of wall segments
    Returns: array of shape (N, 2, 2) where:
        N = number of walls
        First 2 = start/end point
        Second 2 = x,y coordinates
    """
    # Convert to array and reshape to (N,3) where columns are x,y,timestamp
    points = jnp.array(walls_list).reshape(-1, 3)
    # Get consecutive pairs of points
    p1 = points[:-1]  # All points except last
    p2 = points[1:]   # All points except first
    # Keep only pairs with matching timestamps
    mask = p1[:, 2] == p2[:, 2]
    # Stack the x,y coordinates into wall segments
    segments = jnp.stack([p1[mask][:, :2], p2[mask][:, :2]], axis=1)
    return segments

def create_reality(walls_list: List[List[float]], motion_noise: float, sensor_noise: float) -> reality.Reality:
    """Create Reality instance with proper JAX arrays"""
    walls = convert_walls_to_jax(walls_list)
    return reality.Reality(walls, motion_noise, sensor_noise)

def path_to_controls(path_points: List[List[float]]) -> List[Tuple[float, float]]:
    """Convert a series of points into (distance, angle) control pairs
    Returns: List of (forward_dist, rotation_angle) controls
    """
    controls = []
    for i in range(len(path_points) - 1):
        p1 = jnp.array(path_points[i][:2])  # current point [x,y]
        p2 = jnp.array(path_points[i+1][:2])  # next point [x,y]
        
        # Calculate distance and angle to next point
        delta = p2 - p1
        distance = jnp.linalg.norm(delta)
        target_angle = jnp.arctan2(delta[1], delta[0])
        
        # If not first point, need to rotate from previous heading
        if i > 0:
            prev_delta = p1 - jnp.array(path_points[i-1][:2])
            prev_angle = jnp.arctan2(prev_delta[1], prev_delta[0])
            rotation = target_angle - prev_angle
        else:
            # For first point, rotate from initial heading (0)
            rotation = target_angle
            
        controls.append((float(distance), float(rotation)))
    
    return controls

def debug_reality(widget, e):
    """Quick visual check of Reality class"""
    if not widget.state.robot_path:
        return  # Need a path to get initial pose
        
    # Get initial pose from start of path
    start_point = widget.state.robot_path[0]
    initial_pose = Pose(jnp.array([start_point[0], start_point[1]]), 0.0)
    
    walls = convert_walls_to_jax(widget.state.walls)
    world = reality.Reality(walls, 
                   motion_noise=widget.state.motion_noise,
                   sensor_noise=widget.state.sensor_noise,
                   initial_pose=initial_pose)
    
    # Convert path to controls and execute them
    controls = path_to_controls(widget.state.robot_path)
    readings = []
    true_path = [[float(world._true_pose.p[0]), float(world._true_pose.p[1])]]  # Start position
    
    for control in controls:
        reading = world.execute_control(control)
        readings.append(reading)
        # Record position after each control
        true_path.append([float(world._true_pose.p[0]), float(world._true_pose.p[1])])
    
    # Generate possible paths
    key = jax.random.PRNGKey(0)
    possible_paths = sample_possible_paths(key, 20, widget.state.robot_path, widget.state.walls)
        
    
    # Update state with final readings, pose, and full path
    widget.state.update({
        "robot_pose": {
            "x": float(world._true_pose.p[0]),
            "y": float(world._true_pose.p[1]),
            "heading": float(world._true_pose.hd)
        },
        "possible_paths": possible_paths,
        "sensor_readings": readings[-1] if readings else [],
        "true_path": true_path,
        "show_debug": True
    })
    
def clear_state(w, _):
    w.state.update(initial_state)
    
        

# Add debug button to toolbar
toolbar = Plot.html("Select tool:") | ["div", {"class": "flex gap-2 h-10"},
            ["button", {
                "class": js("$state.selected_tool === 'walls' ? 'px-3 py-1 rounded bg-gray-300 hover:bg-gray-400 active:bg-gray-500 focus:outline-none' : 'px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 active:bg-gray-400 focus:outline-none'"),
                "onClick": js("() => $state.selected_tool = 'walls'")
            }, "Draw Walls"],
            ["button", {
                "class": js("$state.selected_tool === 'path' ? 'px-3 py-1 rounded bg-gray-300 hover:bg-gray-400 active:bg-gray-500 focus:outline-none' : 'px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 active:bg-gray-400 focus:outline-none'"),
                "onClick": js("() => $state.selected_tool = 'path'")
            }, "Draw Robot Path"],
            ["button", {
                "class": "px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 active:bg-gray-400",
                "onClick": clear_state
            }, "Clear"]
        ]


reality_toggle = Plot.html("") | ["label.flex.items-center.gap-2.p-2", 
                                  "Show true position:",
                                  ["input", {
        "type": "checkbox", 
        "checked": js("$state.show_true_position"),
        "onChange": js("(e) => $state.show_true_position = e.target.checked")
    }]]

# Modify the onChange handlers at the bottom
(
    canvas & 
    (toolbar | sliders | reality_toggle | sensor_rays + {"height": 200}) 
    | Plot.initialState(initial_state, sync=True)
    | Plot.onChange({
        "robot_path": debug_reality,
        "sensor_noise": debug_reality,
        "motion_noise": debug_reality,
        "walls": debug_reality
    }))
