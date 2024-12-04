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

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Any, Dict

import genjax
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey, split
from penzai import pz
from genstudio.plot import js

import robot_2.emoji as emoji
import robot_2.visualization as v

key = PRNGKey(0)


WALL_COLLISION_THRESHOLD = 0.15
WALL_WIDTH = 6
PATH_WIDTH = 6


@pz.pytree_dataclass
class Pose(genjax.PythonicPytree):
    """Robot pose with position and heading"""

    p: jax.numpy.ndarray  # [x, y]
    hd: float  # heading in radians

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
class World(genjax.PythonicPytree):
    """The physical environment with walls that robots can collide with"""

    walls: jnp.ndarray  # [N, 2, 2] array of wall segments

    @jax.jit
    def ray_distance(
        self, ray_start: jnp.ndarray, ray_dir: jnp.ndarray, max_dist: float
    ) -> float:
        """Find distance to nearest wall along a ray"""
        if self.walls.shape[0] == 0:  # No walls
            return max_dist

        # Vectorized computation for all walls at once
        p1 = self.walls[:, 0]  # Shape: (N, 2)
        p2 = self.walls[:, 1]  # Shape: (N, 2)

        # Wall direction vectors
        wall_vec = p2 - p1  # Shape: (N, 2)

        # Vector from wall start to ray start
        to_start = ray_start - p1  # Shape: (N, 2)

        # Compute determinant (cross product in 2D)
        det = wall_vec[:, 0] * (-ray_dir[1]) - wall_vec[:, 1] * (-ray_dir[0])

        # Compute intersection parameters
        u = (to_start[:, 0] * (-ray_dir[1]) - to_start[:, 1] * (-ray_dir[0])) / (
            det + 1e-10
        )
        t = (wall_vec[:, 0] * to_start[:, 1] - wall_vec[:, 1] * to_start[:, 0]) / (
            det + 1e-10
        )

        # Valid intersections: not parallel, in front of ray, within wall segment
        is_valid = (jnp.abs(det) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)

        # Find minimum valid distance
        min_dist = jnp.min(jnp.where(is_valid, t * jnp.linalg.norm(ray_dir), jnp.inf))
        return jnp.where(jnp.isinf(min_dist), max_dist, min_dist)

    @jax.jit
    def check_movement(
        self,
        start_pos: jnp.ndarray,
        end_pos: jnp.ndarray,
        collision_radius: float = WALL_COLLISION_THRESHOLD,
    ) -> Tuple[bool, jnp.ndarray]:
        """Check if movement between two points collides with walls

        Args:
            start_pos: [x, y] starting position
            end_pos: [x, y] intended end position
            collision_radius: How close we can get to walls

        Returns:
            (can_move, safe_pos) where safe_pos is either end_pos or the
            furthest safe position along the movement line
        """
        movement_dir = end_pos - start_pos
        dist = jnp.linalg.norm(movement_dir)

        # Replace if with where
        ray_dir = jnp.where(
            dist > 1e-6,
            movement_dir / dist,
            jnp.array([1.0, 0.0]),  # Default direction if no movement
        )

        wall_dist = self.ray_distance(start_pos, ray_dir, dist)

        # Stop short of wall by collision_radius
        safe_dist = jnp.maximum(0.0, wall_dist - collision_radius)
        safe_pos = start_pos + ray_dir * safe_dist

        # Use where to select between start_pos and safe_pos
        final_pos = jnp.where(dist > 1e-6, safe_pos, start_pos)

        return wall_dist > dist - collision_radius, final_pos


@pz.pytree_dataclass
class RobotCapabilities(genjax.PythonicPytree):
    """Physical capabilities and limitations of the robot"""

    p_noise: float  # Position noise (std dev in meters)
    hd_noise: float  # Heading noise (std dev in radians)
    sensor_noise: float  # Sensor noise (std dev in meters)
    n_sensors: int = 8  # Number of distance sensors
    sensor_range: float = 10.0  # Maximum sensor range in meters

    def try_move(
        self,
        world: World,
        current_pos: jnp.ndarray,
        desired_pos: jnp.ndarray,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Try to move to desired_pos, respecting walls and adding noise"""
        # Add motion noise
        noise = jax.random.normal(key, shape=(2,)) * self.p_noise
        noisy_target = desired_pos + noise

        # Check for collisions
        _, safe_pos = world.check_movement(current_pos, noisy_target)
        return safe_pos


def path_to_controls(path_points: List[List[float]]) -> jnp.ndarray:
    """Convert a series of points into (distance, angle) control pairs"""
    points = jnp.array([p[:2] for p in path_points])
    deltas = points[1:] - points[:-1]
    distances = jnp.linalg.norm(deltas, axis=1)
    angles = jnp.arctan2(deltas[:, 1], deltas[:, 0])
    angle_changes = jnp.diff(angles, prepend=0.0)
    return jnp.stack([distances, angle_changes], axis=1)


@jax.jit
def get_sensor_readings(
    world: World, robot: RobotCapabilities, pose: Pose, key: PRNGKey
) -> Tuple[jnp.ndarray, PRNGKey]:
    """Return noisy distance readings to walls from given pose"""
    MAX_SENSORS = 32  # Fixed maximum
    key, subkey = jax.random.split(key)

    # Calculate angles based on n_sensors, but generate MAX_SENSORS of them
    angle_step = 2 * jnp.pi / robot.n_sensors
    angles = jnp.arange(MAX_SENSORS) * angle_step
    noise = jax.random.normal(subkey, (MAX_SENSORS,)) * robot.sensor_noise

    readings = jax.vmap(
        lambda a: world.ray_distance(
            ray_start=pose.p,
            ray_dir=jnp.array([jnp.cos(pose.hd + a), jnp.sin(pose.hd + a)]),
            max_dist=robot.sensor_range,
        )
    )(angles)

    # Create a mask for the first n_sensors elements
    mask = jnp.arange(MAX_SENSORS) < robot.n_sensors

    # Apply mask and pad with zeros
    readings = (readings + noise) * mask

    return readings, key


@jax.jit
def execute_control(
    world: World,
    robot: RobotCapabilities,
    current_pose: Pose,
    control: Tuple[float, float],
    key: PRNGKey,
) -> Tuple[Pose, jnp.ndarray, PRNGKey]:
    """Execute a control command with noise, stopping if we hit a wall

    Args:
        control: (distance, angle) pair where:
            - angle is how much to turn FIRST
            - distance is how far to move AFTER turning
    """
    dist, angle = control
    k1, k2, k3 = jax.random.split(key, 3)

    # Add noise to motion
    noisy_dist = dist + jax.random.normal(k1) * robot.p_noise
    noisy_angle = angle + jax.random.normal(k2) * robot.hd_noise

    # First rotate (can always rotate)
    new_pose = current_pose.rotate(noisy_angle)

    # Check distance to wall in our current heading direction
    min_dist = world.ray_distance(
        ray_start=new_pose.p, ray_dir=new_pose.dp(), max_dist=robot.sensor_range
    )

    # Only move as far as we can before hitting a wall
    safe_dist = jnp.minimum(noisy_dist, min_dist - WALL_COLLISION_THRESHOLD)
    safe_dist = jnp.maximum(safe_dist, 0)  # Don't move backwards

    new_pose = new_pose.step_along(safe_dist)

    # Get sensor readings from new position
    readings, k4 = get_sensor_readings(world, robot, new_pose, k3)

    return new_pose, readings, k4


@jax.jit
def simulate_robot_path(
    world: World,
    robot: RobotCapabilities,
    start_pose: Pose,
    controls: jnp.ndarray,
    key: jnp.ndarray,
):
    """Simulate robot path with noise and sensor readings

    Returns:
        Tuple of:
        - Array of shape [n_steps, 2] containing positions
        - Array of shape [n_steps] containing headings
        - Array of shape [n_steps, n_sensors] containing sensor readings
    """

    def step_fn(carry, control):
        pose, k = carry
        new_pose, readings, new_key = execute_control(
            world=world, robot=robot, current_pose=pose, control=control, key=k
        )
        return (new_pose, new_key), (new_pose, readings)

    (_, _), (poses, readings) = jax.lax.scan(step_fn, (start_pose, key), controls)

    # Extract positions and headings
    positions = jnp.concatenate([start_pose.p[None, :], jax.vmap(lambda p: p.p)(poses)])
    headings = jnp.concatenate(
        [jnp.array([start_pose.hd]), jax.vmap(lambda p: p.hd)(poses)]
    )

    return positions, headings, readings


@partial(jax.jit, static_argnums=(1))
def sample_possible_paths(
    key: jnp.ndarray,
    n_paths: int,
    robot_path: jnp.ndarray,
    world: World,
    robot: RobotCapabilities,
):
    """Generate n possible paths given the planned path, respecting walls"""
    path_points = robot_path[:, :2]
    controls = path_to_controls(path_points)

    start_point = path_points[0]
    start_pose = Pose(jnp.array(start_point, dtype=jnp.float32), 0.0)

    keys = jax.random.split(key, n_paths)

    # Vectorize over different random keys
    return jax.vmap(
        lambda k: simulate_robot_path(
            world=world, robot=robot, start_pose=start_pose, controls=controls, key=k
        )
    )(keys)


def walls_to_jax(walls_list: List[List[float]]) -> jnp.ndarray:
    """Convert wall vertices from UI format to JAX array of wall segments"""
    if not walls_list:
        return jnp.array([]).reshape((0, 2, 2))

    points = jnp.array(walls_list, dtype=jnp.float32)
    p1 = points[:-1]
    p2 = points[1:]

    segments = jnp.stack([p1[:, :2], p2[:, :2]], axis=1)

    valid_mask = p1[:, 2] == p2[:, 2]
    return segments * valid_mask[:, None, None]


def simulate_robot_uncertainty(widget, e, seed=None):
    """Handle updates to robot simulation"""
    if not widget.state.robot_path:
        return

    current_seed = jnp.array(seed if seed is not None else widget.state.current_seed)
    assert jnp.issubdtype(current_seed.dtype, jnp.integer), "Seed must be an integer"

    current_key = PRNGKey(current_seed)

    # Create world and robot objects
    world = World(walls_to_jax(widget.state.walls))
    robot = RobotCapabilities(
        p_noise=widget.state.motion_noise,
        hd_noise=widget.state.motion_noise * widget.state.heading_noise_scale,
        sensor_noise=widget.state.sensor_noise,
        n_sensors=widget.state.n_sensors,
        sensor_range=10.0,
    )

    path = jnp.array(widget.state.robot_path, dtype=jnp.float32)

    # Sample all paths at once (1 true path + N possible paths)
    n_possible = 40
    all_paths, all_headings, all_readings = sample_possible_paths(
        current_key, n_possible + 1, path, world, robot
    )

    # First path is the "true" path
    true_path = all_paths[0]
    final_readings = all_readings[0, -1]
    final_heading = all_headings[0, -1]

    # Remaining paths are possible paths
    possible_paths = all_paths[1:]

    widget.state.update(
        {
            "robot_pose": {
                "x": float(true_path[-1, 0]),
                "y": float(true_path[-1, 1]),
                "heading": float(final_heading),
            },
            "possible_paths": possible_paths,
            "sensor_readings": final_readings,
            "true_path": [[float(x), float(y)] for x, y in true_path],
            "show_debug": True,
            "current_seed": current_seed,
        }
    )


drawing_system_handler = Plot.js("""({points, simplify}) => {
        mode = $state.selected_tool
        if (mode === 'walls') {
            $state.update(['walls', 'concat', simplify(0.25)])
        }
        if (mode === 'path') {
            $state.robot_path = simplify(0.5)
        }
    }""")

sliders = Plot.Slider(
    "motion_noise", range=[0, 0.5], step=0.01, label="Motion Noise:", showValue=True
) & Plot.Slider(
    "heading_noise_scale",
    range=[0, 1],
    step=0.05,
    label="Heading Noise Scale:",
    showValue=True,
) | Plot.Slider(
    "sensor_noise", range=[0, 1], step=0.02, label="Sensor Noise:", showValue=True
) & Plot.Slider("n_sensors", range=[4, 32], step=1, label="Sensors:", showValue=True)


def create_initial_state(seed) -> Dict[str, Any]:
    """Create initial state for visualization"""
    return {
        "walls": [
            # Frame around domain (timestamp 0)
            [0, 0, 0],
            [10, 0, 0],  # Bottom
            [10, 0, 0],
            [10, 10, 0],  # Right
            [10, 10, 0],
            [0, 10, 0],  # Top
            [0, 10, 0],
            [0, 0, 0],  # Left
        ],
        "robot_pose": {"x": 0.5, "y": 0.5, "heading": 0},
        "sensor_noise": 0.1,
        "motion_noise": 0.1,
        "heading_noise_scale": 0.3,
        "n_sensors": 8,
        "show_sensors": True,
        "selected_tool": "path",
        "robot_path": [],
        "possible_paths": [],
        "estimated_pose": None,
        "sensor_readings": [],
        "sensor_explore_angle": -1,
        "show_uncertainty": True,
        "show_true_position": False,
        "current_line": [],
        "current_seed": seed,
    }


true_position_toggle = Plot.html(
    [
        "label.flex.items-center.gap-2.p-2.bg-gray-100.rounded.hover:bg-gray-300",
        [
            "input",
            {
                "type": "checkbox",
                "checked": js("$state.show_true_position"),
                "onChange": js("(e) => $state.show_true_position = e.target.checked"),
            },
        ],
        "Show true position",
    ]
)

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
    marker="circle",
)


rotating_sensor_rays = (
    Plot.line(
        js("""
           Array.from($state.sensor_readings).map((r, i) => {
            const heading = $state.robot_pose.heading || 0;
            const n_sensors = $state.n_sensors;
            let angle = heading + (i * Math.PI * 2) / n_sensors;
            if (!$state.show_true_position) {
                if ($state.sensor_explore_angle > -1) {
                    angle += $state.sensor_explore_angle
                } else {
                    angle += $state.current_seed || Math.random() * 2 * Math.PI;
                }
            }
            const x = $state.robot_pose.x;
            const y = $state.robot_pose.y;
            return [
                [0, 0, i],
                [r * Math.cos(angle),
                 r * Math.sin(angle), i]
            ]
           }).flat()
           """),
        z="2",
        stroke="red",
        strokeWidth=1,
        marker="circle",
    )
    # move the mouse around the plot to rotate the sensors
    + Plot.events(
        {
            "onMouseMove": Plot.js("""(e) => {
                    // Convert mouse position to angle from center
                    // atan2 gives angle in radians from -pi to pi
                    // Subtract pi/2 to make 12 o'clock 0 radians
                    const angle = Math.atan2(e.y, e.x) - Math.PI/2;

                    // Normalize to 0 to 2pi range
                    const normalized = (angle + 2*Math.PI) % (2*Math.PI);

                    $state.sensor_explore_angle = normalized;
                }""")
        }
    )
    + {"height": 200, "width": 200, "className": "bg-gray-100"}
    + Plot.aspectRatio(1)
    + Plot.domain([-10, 10])
    + Plot.hideAxis()
    + Plot.gridX(interval=1)
    + Plot.gridY(interval=1)
)

true_path = Plot.cond(
    js("$state.show_true_position"),
    [
        Plot.line(
            js("$state.true_path"), stroke=Plot.constantly("True Path"), strokeWidth=2
        ),
        sensor_rays,
        Plot.text(
            js("[[$state.robot_pose.x, $state.robot_pose.y]]"),
            text=Plot.constantly(emoji.robot),
            fontSize=30,
            textAnchor="middle",
            dy="-0.35em",
            rotate=js("(-$state.robot_pose.heading + Math.PI/2) * 180 / Math.PI"),
        ),
    ],
)

planned_path = Plot.line(
    js("$state.robot_path"),
    stroke=Plot.constantly("Robot Path"),
    strokeWidth=2,
    r=3,
    marker="circle",
)

walls = Plot.line(
    js("$state.walls"),
    stroke=Plot.constantly("Walls"),
    strokeWidth=WALL_WIDTH,
    z="2",
    render=Plot.renderChildEvents(
        {
            "onClick": js("""(e) => {
                const zs = new Set($state.walls.map(w => w[2]));
                const targetZ = [...zs][e.index];
                $state.walls = $state.walls.filter(([x, y, z]) => z !== targetZ)
                }""")
        }
    ),
)

possible_paths = Plot.line(
    js(
        """
               if (!$state.show_debug || !$state.possible_paths) {return [];};
               return $state.possible_paths.flatMap((path, pathIdx) =>
                   path.map(([x, y]) => [x, y, pathIdx])
               )
            """,
        expression=False,
    ),
    stroke="blue",
    strokeOpacity=0.2,
    z="2",
)


def clear_state(w, _):
    """Reset visualization state"""
    w.state.update(
        create_initial_state(w.state.current_seed)
        | {"selected_tool": w.state.selected_tool}
    )


selectable_button = "button.px-3.py-1.rounded.bg-gray-100.hover:bg-gray-300.data-[selected=true]:bg-gray-300"

toolbar = Plot.html("Select tool:") | [
    "div.flex.gap-2",
    [
        selectable_button,
        {
            "data-selected": js("$state.selected_tool === 'path'"),
            "onClick": js("() => $state.selected_tool = 'path'"),
        },
        f"{emoji.robot} Path",
    ],
    [
        selectable_button,
        {
            "data-selected": js("$state.selected_tool === 'walls'"),
            "onClick": js("() => $state.selected_tool = 'walls'"),
        },
        f"{emoji.pencil} Walls",
    ],
    [selectable_button, {"onClick": clear_state}, "Clear"],
]


def handleSeedIndex(w, e):
    global key
    try:
        if e.index == 0:
            seed = key[0]
        elif e.index == -1:
            key = split(key, 2)[0]
            seed = key[0]
        else:
            seed = split(key, e.index)[e.index - 1][0]
        simulate_robot_uncertainty(w, e, seed=seed)
    except Exception as err:
        print(f"Error handling seed index: {err}, {e.key}, {e.index}")


key_scrubber = v.key_scrubber(handleSeedIndex)

canvas = (
    v.drawing_system("current_line", drawing_system_handler)
    + walls
    + planned_path
    + possible_paths
    + true_path
    + Plot.domain([0, 10], [0, 10])
    + Plot.grid()
    + Plot.aspectRatio(1)
    + Plot.colorMap(
        {
            "Walls": "#666",
            "Sensor Rays": "red",
            "True Path": "green",
            "Robot Path": "blue",
        }
    )
    + Plot.colorLegend()
    + Plot.clip()
    + Plot.gridX(interval=1)
)

(
    canvas
    & (sliders | toolbar | true_position_toggle | key_scrubber | rotating_sensor_rays)
    & {"widths": ["400px", 1]}
    | Plot.initialState(create_initial_state(0), sync=True)
    | Plot.onChange(
        {
            "robot_path": simulate_robot_uncertainty,
            "sensor_noise": simulate_robot_uncertainty,
            "motion_noise": simulate_robot_uncertainty,
            "heading_noise_scale": simulate_robot_uncertainty,
            "n_sensors": simulate_robot_uncertainty,
            "walls": simulate_robot_uncertainty,
        }
    )
)
