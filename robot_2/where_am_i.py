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
from robot_2.reality import Pose
from functools import partial

WALL_WIDTH=6
PATH_WIDTH=6
SEGMENT_THRESHOLD=0.25
@pz.pytree_dataclass
class RobotSettings(genjax.PythonicPytree):
    """Robot configuration and uncertainty settings"""
    p_noise: float = 0.1        # Position noise
    hd_noise: float = 0.1       # Heading noise
    sensor_noise: float = 0.1   # Sensor noise
    sensor_range: float = 10.0  # Maximum sensor range

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

def sample_single_path(carry, control, walls, n_sensors, settings):
    """Single step of path sampling that can be used with scan"""
    pose, key = carry
    pose, _, key = reality.execute_control(
        walls=walls,
        n_sensors=n_sensors,
        settings=settings,
        current_pose=pose,
        control=control,
        key=key
    )
    return (pose, key), pose.p

@partial(jax.jit, static_argnums=(1))  # n_paths is static
def sample_possible_paths(key: jax.random.PRNGKey, n_paths: int, n_sensors: int,
                         robot_path: jnp.ndarray, walls: jnp.ndarray, 
                         settings: RobotSettings):
    """Generate n possible paths given the planned path, respecting walls"""
    # Extract just x,y coordinates from path
    path_points = robot_path[:, :2]  # Shape: (N, 2)
    controls = path_to_controls(path_points)
    
    start_point = path_points[0]
    start_pose = Pose(jnp.array(start_point, dtype=jnp.float32), 0.0)
    
    # Split key for multiple samples
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

_gensym_counter = 0


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
                if (Math.sqrt(dx*dx + dy*dy) >= {SEGMENT_THRESHOLD}) {{
                    $state.update(['{key}', 'append', [e.x, e.y, e.startTime]]);
                }}
            }}
        }}"""),
        "onDrawEnd": js(f"""(e) => {{
            if ($state.{key}.length > 1) {{
                const points = [...$state.{key}, [e.x, e.y, e.startTime]] 
                
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
            step=0.02,
            label="Sensor Noise:", 
            showValue=True
        )
     | Plot.Slider(
            "motion_noise",
            range=[0, 1],
            step=0.02,
            label="Motion Noise:",
            showValue=True
        )
     | Plot.Slider(
            "n_sensors",
            range=[4, 32],
            step=1,
            label="Number of Sensors:",
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
        "n_sensors": 8,
        "show_sensors": True,
        "selected_tool": "path",
        "robot_path": [],
        "possible_paths": [],
        "estimated_pose": None,
        "sensor_readings": [],
        "show_uncertainty": True,
        "show_true_position": False
    }

sensor_rays = Plot.line(
    js("""
       Array.from($state.sensor_readings).map((r, i) => {
        const heading = $state.robot_pose.heading || 0;
        const n_sensors = $state.n_sensors;
        const angle = heading + (i * Math.PI * 2) / n_sensors;
        const x = $state.robot_pose.x;
        const y = $state.robot_pose.y;
        return [
            [x, y, i],
            [x + r * Math.cos(angle), 
             y + r * Math.sin(angle), i]
        ]
       }).flat()
       """),
    z="2",
    stroke="red",
    strokeWidth=1,
    marker="circle"
)

true_path = Plot.line(
                js("$state.true_path"),
                stroke=Plot.constantly("True Path"),
                strokeWidth=2
            )

planned_path = Plot.line(
                    js("$state.robot_path"),
                    stroke=Plot.constantly("Robot Path"),
                    strokeWidth=2,
                    r=3,
                    marker="circle"),

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
                    rotate=js("(-$state.robot_pose.heading + Math.PI/2) * 180 / Math.PI")), 
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
        strokeOpacity=0.2,
        z="2"
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
    if not walls_list:
        return jnp.array([]).reshape((0, 2, 2))
    
    # Convert everything to JAX at once, using float32 for timestamps
    points = jnp.array(walls_list, dtype=jnp.float32)  # Shape: (N, 3)
    
    # Get consecutive pairs of points
    p1 = points[:-1]   # Shape: (N-1, 3)
    p2 = points[1:]    # Shape: (N-1, 3)
    
    # Create wall segments array
    segments = jnp.stack([
        p1[:, :2],     # x,y coordinates of start points
        p2[:, :2]      # x,y coordinates of end points
    ], axis=1)         # Shape: (N-1, 2, 2)
    
    # Use timestamps to mask valid segments
    valid_mask = p1[:, 2] == p2[:, 2]
    
    # Return masked segments
    return segments * valid_mask[:, None, None]

def path_to_controls(path_points: List[List[float]]) -> jnp.ndarray:
    """Convert a series of points into (distance, angle) control pairs
    Returns: JAX array of shape (N,2) containing (forward_dist, rotation_angle) controls
    """
    points = jnp.array([p[:2] for p in path_points])
    deltas = points[1:] - points[:-1]
    distances = jnp.linalg.norm(deltas, axis=1)
    angles = jnp.arctan2(deltas[:, 1], deltas[:, 0])
    # Calculate angle changes
    angle_changes = jnp.diff(angles, prepend=0.0)
    return jnp.stack([distances, angle_changes], axis=1)

@jax.jit
def simulate_robot_path(start_pose: Pose, n_sensors: int, controls: jnp.ndarray, 
                       walls: jnp.ndarray, settings: RobotSettings, key: jax.random.PRNGKey):
    """Jitted pure function for simulating robot path"""
    def step_fn(carry, control):
        pose, k = carry
        new_pose, readings, new_key = reality.execute_control(
            walls=walls,
            n_sensors=n_sensors, 
            settings=settings,
            current_pose=pose,
            control=control,
            key=k
        )
        return (new_pose, new_key), (new_pose, readings)
    
    return jax.lax.scan(step_fn, (start_pose, key), controls)

def debug_reality(widget, e):
    if not widget.state.robot_path:
        return
        
    # Create settings object
    settings = RobotSettings(
        p_noise=widget.state.motion_noise,
        hd_noise=widget.state.motion_noise,
        sensor_noise=widget.state.sensor_noise,
    )
    
    # Handle data conversion at the boundary
    path = jnp.array(widget.state.robot_path, dtype=jnp.float32)
    walls = convert_walls_to_jax(widget.state.walls)
    n_sensors = int(widget.state.n_sensors)  # Convert to int explicitly
    
    start_pose = Pose(path[0, :2], 0.0)
    controls = path_to_controls(path)
    key = jax.random.PRNGKey(0)
    
    # Use jitted function for core computation
    (final_pose, _), (poses, readings) = simulate_robot_path(
        start_pose, n_sensors, controls, walls, settings, key
    )
    
    # Convert poses to path
    true_path = jnp.concatenate([start_pose.p[None, :], jax.vmap(lambda p: p.p)(poses)])
    
    # Generate possible paths
    possible_paths = sample_possible_paths(
        key, 20, n_sensors, path, walls, settings  # Pass n_sensors separately
    )
    
    # Update widget state
    widget.state.update({
        "robot_pose": {
            "x": float(final_pose.p[0]),
            "y": float(final_pose.p[1]),
            "heading": float(final_pose.hd)
        },
        "possible_paths": possible_paths,
        "sensor_readings": readings[-1] if len(readings) > 0 else [],
        "true_path": [[float(x), float(y)] for x, y in true_path],
        "show_debug": True
    })

def clear_state(w, _):
    w.state.update(initial_state | {"selected_tool": w.state.selected_tool})
    

selectable_button = "button.px-3.py-1.rounded.bg-gray-100.hover:bg-gray-300.data-[selected=true]:bg-gray-300"
        
# Add debug button to toolbar
toolbar = Plot.html("Select tool:") | ["div.flex.gap-2",
            [selectable_button, {
                "data-selected": js("$state.selected_tool === 'path'"),
                "onClick": js("() => $state.selected_tool = 'path'")
            }, "🤖 Path"],
            [selectable_button, {
                "data-selected": js("$state.selected_tool === 'walls'"),
                "onClick": js("() => $state.selected_tool = 'walls'")
            }, "✏️ Walls"],
            [selectable_button, {
                "onClick": clear_state
            }, "Clear"]
        ]


reality_toggle = Plot.html("") | ["label.flex.items-center.gap-2.p-2.bg-gray-100.rounded.hover:bg-gray-300", 
                                  ["input", {
        "type": "checkbox", 
        "checked": js("$state.show_true_position"),
        "onChange": js("(e) => $state.show_true_position = e.target.checked")
    }], "Show true position"]

# Modify the onChange handlers at the bottom
(
    canvas & 
    (sliders | toolbar | reality_toggle | sensor_rays + {"height": 200}) 
    & {"widths": ["400px", 1]}
    | Plot.initialState(initial_state, sync=True)
    | Plot.onChange({
        "robot_path": debug_reality,
        "sensor_noise": debug_reality,
        "motion_noise": debug_reality,
        "n_sensors": debug_reality,
        "walls": debug_reality
    }))
