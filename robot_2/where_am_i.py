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
#    - Black line: Where the robot figures it ACTUALLY is

# %%
# pyright: reportUnusedExpression=false
# pyright: reportUnknownMemberType=false

from typing import List, Tuple, Any, Dict

import genjax
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import time
from jax.random import PRNGKey, split
from penzai import pz
from genstudio.plot import js
from functools import partial

import robot_2.emoji as emoji
import robot_2.visualization as v

key = PRNGKey(0)

WALL_COLLISION_THRESHOLD = jnp.array(0.15)
WALL_BOUNCE = 0.15
MAX_SENSORS = 32


@pz.pytree_dataclass
class Pose(genjax.PythonicPytree):
    """Robot pose with position and heading"""

    p: jnp.ndarray  # [x, y]
    hd: jnp.ndarray  # heading in radians

    def dp(self):
        """Get direction vector from heading"""
        return jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])

    def step_along(self, s: jnp.ndarray) -> "Pose":
        """Move forward by distance s"""
        return Pose(self.p + s * self.dp(), self.hd)

    def rotate(self, angle: jnp.ndarray) -> "Pose":
        """Rotate by angle (in radians)"""
        return Pose(self.p, self.hd + angle)

    def for_json(self):
        if len(self.p.shape) == 1:
            return [*self.p, self.hd]
        heading_expanded = jnp.expand_dims(self.hd, axis=-1)  # Add last dimension
        return jnp.concatenate([self.p, heading_expanded], axis=-1)


def calculate_bounce_point(
    collision_point: jnp.ndarray,
    ray_dir: jnp.ndarray,
    wall_vec: jnp.ndarray,
    bounce_amount: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate bounce point for a single wall collision

    Args:
        collision_point: Point of collision with wall
        ray_dir: Direction of incoming ray
        wall_vec: Vector along wall direction
        bounce_amount: How far to bounce

    Returns:
        Point after bouncing off wall
    """
    wall_normal = jnp.array([-wall_vec[1], wall_vec[0]]) / (
        jnp.linalg.norm(wall_vec) + 1e-10
    )
    # Ensure wall normal points away from approach direction
    wall_normal = jnp.where(
        jnp.dot(ray_dir, wall_normal) > 0, -wall_normal, wall_normal
    )
    return collision_point + bounce_amount * wall_normal


def compute_wall_normal(wall_direction: jnp.ndarray) -> jnp.ndarray:
    """Compute unit normal vector to wall direction"""
    return jnp.array([-wall_direction[1], wall_direction[0]]) / (
        jnp.linalg.norm(wall_direction) + 1e-10
    )


@pz.pytree_dataclass
class World(genjax.PythonicPytree):
    """The physical environment with walls that robots can collide with"""

    walls: jnp.ndarray  # [N, 2, 2] array of wall segments
    wall_vecs: jnp.ndarray  # [N, 2] array of wall direction vectors
    bounce: jnp.ndarray = WALL_BOUNCE  # How much to bounce off walls
    __hash__ = None

    def physical_step(
        self, start_pos: jnp.ndarray, end_pos: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute physical step with wall collisions and bounces

        Args:
            start_pos: Starting position [x, y]
            end_pos: Intended end position [x, y]
            heading: Current heading in radians

        Returns:
            New pose after movement, considering wall collisions
        """
        # Calculate step properties
        step_direction = end_pos - start_pos
        step_length = jnp.linalg.norm(step_direction)

        # Get distance to nearest wall
        ray_dir = step_direction / (step_length + 1e-10)  # Avoid division by zero
        wall_dist, wall_idx = self.ray_distance(start_pos, ray_dir, step_length)

        # Find collision point
        collision_point = start_pos + ray_dir * wall_dist

        # Calculate bounce point if wall hit
        bounce_pos = calculate_bounce_point(
            collision_point, ray_dir, self.wall_vecs[wall_idx], self.bounce
        )

        # Define conditions for position selection
        conditions = [
            step_length < 1e-6,  # No movement case
            wall_dist >= step_length,  # No collision case
            wall_idx >= 0,  # Wall collision case
        ]

        positions = [
            start_pos,  # For no movement
            end_pos,  # For no collision
            bounce_pos,  # For wall collision
        ]

        final_pos = jnp.select(conditions, positions, default=end_pos)

        return final_pos

    def ray_distance(
        self, ray_start: jnp.ndarray, ray_dir: jnp.ndarray, max_dist: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Find distance to nearest wall along a ray and which wall was hit

        Args:
            ray_start: Starting point of ray
            ray_dir: Direction of ray (normalized)
            max_dist: Maximum distance to check

        Returns:
            Tuple of (distance, wall_idx) where:
            - distance: Distance to nearest wall, or max_dist + threshold if no wall hit
            - wall_idx: Index of wall that was hit, or -1 if no wall hit
        """
        if self.walls.shape[0] == 0:  # No walls
            return max_dist + WALL_COLLISION_THRESHOLD, jnp.array(-1)

        # Vectorized computation for all walls at once
        p1 = self.walls[:, 0]  # Shape: (N, 2)

        # Vector from wall start to ray start
        to_start = ray_start - p1  # Shape: (N, 2)

        # Compute determinant (cross product in 2D) using pre-computed wall vectors
        det = self.wall_vecs[:, 0] * (-ray_dir[1]) - self.wall_vecs[:, 1] * (
            -ray_dir[0]
        )

        # Compute intersection parameters
        u = (to_start[:, 0] * (-ray_dir[1]) - to_start[:, 1] * (-ray_dir[0])) / (
            det + 1e-10
        )
        t = (
            self.wall_vecs[:, 0] * to_start[:, 1]
            - self.wall_vecs[:, 1] * to_start[:, 0]
        ) / (det + 1e-10)

        # Valid intersections: not parallel, in front of ray, within wall segment
        is_valid = (jnp.abs(det) > 1e-10) & (t >= 0) & (u >= 0) & (u <= 1)
        distances = jnp.where(is_valid, t * jnp.linalg.norm(ray_dir), jnp.inf)

        # Find closest valid wall
        closest_idx = jnp.argmin(distances)
        min_dist = distances[closest_idx]

        # Return -1 as wall index if no valid intersection found
        wall_idx = jnp.where(jnp.isinf(min_dist), -1, closest_idx)
        final_dist = jnp.where(
            jnp.isinf(min_dist), max_dist + WALL_COLLISION_THRESHOLD, min_dist
        )

        return final_dist, wall_idx


def walls_to_jax(walls_list: List[List[float]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert wall vertices from UI format to JAX arrays of wall segments and direction vectors"""
    if not walls_list:
        empty = jnp.array([]).reshape((0, 2, 2))
        return empty, jnp.array([]).reshape((0, 2))

    points = jnp.array(walls_list, dtype=jnp.float32)
    p1 = points[:-1]
    p2 = points[1:]

    segments = jnp.stack([p1[:, :2], p2[:, :2]], axis=1)
    valid_mask = p1[:, 2] == p2[:, 2]

    # Compute wall direction vectors
    wall_segments = segments * valid_mask[:, None, None]
    wall_vecs = (wall_segments[:, 1] - wall_segments[:, 0]) * valid_mask[:, None]

    return wall_segments, wall_vecs


@pz.pytree_dataclass
class RobotCapabilities(genjax.PythonicPytree):
    """Physical capabilities and limitations of the robot"""

    p_noise: jnp.ndarray  # Position noise (std dev in meters)
    hd_noise: jnp.ndarray  # Heading noise (std dev in radians)
    sensor_noise: jnp.ndarray  # Sensor noise (std dev in meters)
    n_sensors: jnp.ndarray = 8  # Number of distance sensors
    sensor_range: jnp.ndarray = 10.0  # Maximum sensor range in meters


def path_to_controls(path_points: jnp.ndarray) -> Tuple[Pose, jnp.ndarray]:
    """Convert a series of points into a starting pose and list of (distance, angle) control pairs

    Each control pair is (distance, angle) where:
    - distance: move this far in current heading
    - angle: after moving, turn this much (relative to current heading)
    """
    points = jnp.array([p[:2] for p in path_points])
    deltas = points[1:] - points[:-1]
    distances = jnp.linalg.norm(deltas, axis=1)

    # Calculate absolute angles for each segment
    angles = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    # Start facing the first segment
    start_pose = Pose(p=points[0], hd=angles[0])

    # For each segment, we need:
    # - distance: length of current segment
    # - angle: turn needed after this segment to face next segment
    angle_changes = jnp.diff(angles, append=angles[-1])

    controls = jnp.stack([distances, angle_changes], axis=1)

    return start_pose, controls


@genjax.gen
def get_sensor_reading(
    world: World, robot: RobotCapabilities, pose: Pose, angle: jnp.ndarray
) -> jnp.ndarray:
    """Get a single noisy sensor reading at the given angle relative to robot heading"""
    # Get the ray direction vector for this sensor angle
    ray_dir = jnp.array([jnp.cos(pose.hd + angle), jnp.sin(pose.hd + angle)])

    # Get raw distance reading
    distance, idx = world.ray_distance(
        ray_start=pose.p, ray_dir=ray_dir, max_dist=robot.sensor_range
    )

    # Add noise to reading
    noisy_distance = genjax.normal(distance, robot.sensor_noise) @ "reading"

    return noisy_distance


@genjax.gen
def execute_control(
    world: World,
    robot: RobotCapabilities,
    current_pose: Pose,
    control: jnp.ndarray,
):
    """Execute a control command with physical step and noise"""
    dist, angle = control

    # Calculate noisy intended position
    planned_pos = current_pose.p + dist * current_pose.dp()
    noisy_pos = (
        genjax.mv_normal_diag(planned_pos, robot.p_noise * jnp.ones(2)) @ "p_noise"
    )
    noisy_angle = genjax.normal(current_pose.hd + angle, robot.hd_noise) @ "hd_noise"
    physical_pos = world.physical_step(current_pose.p, noisy_pos)

    final_pose = Pose(p=physical_pos, hd=noisy_angle)

    sensor_angles = jnp.arange(MAX_SENSORS) * 2 * jnp.pi / robot.n_sensors
    sensor_mask = jnp.arange(MAX_SENSORS) < robot.n_sensors
    get_readings = (
        get_sensor_reading.partial_apply(world, robot, final_pose).mask().vmap()
    )
    readings = get_readings(sensor_mask, sensor_angles) @ "sensor readings"

    return final_pose, (final_pose, readings.value)


@genjax.gen
def sample_robot_path(
    world: World, robot: RobotCapabilities, start: Pose, controls: jnp.ndarray
):
    """Simulate robot path with noise and sensor readings using genjax

    Args:
        world: World containing walls
        robot: Robot capabilities and noise parameters
        start: Starting pose
        controls: Array of (distance, angle) control pairs

    Returns:
        Tuple of:
        - Array of poses for each step (including start pose)
        - Array of sensor readings for each step
    """
    # Prepend a no-op control to get initial readings
    noop = jnp.array([0.0, 0.0])
    all_controls = jnp.concatenate([noop[None], controls])

    # Use execute_control.scan() to process all controls
    _, (path, readings) = (
        execute_control.partial_apply(world, robot).scan()(start, all_controls)
        @ "trajectory"
    )

    return path, readings


def sample_possible_paths(
    world: World, robot: RobotCapabilities, planned_path: jnp.ndarray, n_paths: int, key
):
    """Generate n possible paths given the planned path, respecting walls"""
    (start_pose, controls) = path_to_controls(planned_path[:, :2])
    # Create n random keys for parallel simulation
    keys = jax.random.split(key, n_paths)

    # Vectorize simulation across keys
    return jax.vmap(sample_robot_path.simulate, in_axes=(0, None))(
        keys, (world, robot, start_pose, controls)
    ).get_retval()


@partial(jax.jit, static_argnums=4)
def simulate_robot(
    world: World,
    robot: RobotCapabilities,
    path: jnp.ndarray,
    key: jnp.ndarray,
    n_possible: int = 40,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Core robot simulation logic that can be used for both visualization and benchmarking

    Args:
        world: World containing walls
        robot: Robot capabilities and noise parameters
        path: Array of path points
        key: Random key for simulation
        n_possible: Number of possible paths to simulate (default: 40)

    Returns:
        Tuple of:
        - paths: Array of shape (n_possible + 1, n_steps, 3) containing all simulated paths
        - readings: Array of shape (n_possible + 1, n_steps, n_sensors) containing sensor readings
    """
    # Sample all paths at once (1 true path + N possible paths)
    return sample_possible_paths(world, robot, path, n_possible + 1, key)


def update_robot_simulation(widget, e, seed=None):
    """Handle updates to robot simulation"""
    if not widget.state.robot_path:
        return

    current_seed = jnp.array(seed if seed is not None else widget.state.current_seed)
    assert jnp.issubdtype(current_seed.dtype, jnp.integer), "Seed must be an integer"

    current_key = PRNGKey(current_seed)

    # Create world and robot objects
    world = World(*walls_to_jax(widget.state.walls))
    robot = RobotCapabilities(
        p_noise=jnp.array(widget.state.motion_noise, dtype=jnp.float32),
        hd_noise=jnp.array(
            widget.state.motion_noise * widget.state.heading_noise_scale,
            dtype=jnp.float32,
        ),
        sensor_noise=jnp.array(widget.state.sensor_noise, dtype=jnp.float32),
        n_sensors=jnp.array(widget.state.n_sensors, dtype=jnp.int32),
        sensor_range=jnp.array(10.0, dtype=jnp.float32),
    )

    path = jnp.array(widget.state.robot_path, dtype=jnp.float32)

    # Use the factored out simulation function and measure time
    start_time = time.time()
    paths, readings = simulate_robot(world, robot, path, current_key)
    simulation_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    widget.state.update(
        {
            "possible_paths": paths[1:],
            "true_path": paths[0],
            "robot_readings": readings[0][-1][: robot.n_sensors],
            "show_debug": True,
            "current_seed": current_seed,
            "simulation_time": simulation_time,
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
        "robot_pose": Plot.js("$state.true_path?.[$state.true_path.length-1]"),
        "true_path": None,
        "sensor_noise": 0.1,
        "motion_noise": 0.1,
        "heading_noise_scale": 0.3,
        "n_sensors": 8,
        "selected_tool": "path",
        "robot_path": [],
        "possible_paths": [],
        "estimated_pose": None,
        "robot_readings": None,
        "sensor_explore_angle": -1,
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
    js(
        """
        const readings = $state.robot_readings
        if (!readings) return;
        const n_sensors = readings.length;
        const [x, y, heading] = $state.robot_pose;
        return Array.from($state.robot_readings).flatMap((r, i) => {
            const angle = heading + (i * Math.PI * 2) / n_sensors;
            const from = [x, y, i]
            const to = [x + r * Math.cos(angle), y + r * Math.sin(angle), i]
            return [from, to]
        })
           """,
        expression=False,
    ),
    z="2",
    stroke="red",
    strokeWidth=1,
    marker="circle",
)


rotating_sensor_rays = (
    Plot.line(
        js(
            """
            const readings = $state.robot_readings;
            if (!readings) return;
            const n_sensors = readings.length;
            const [x, y, heading] = $state.robot_pose;

            let angle_modifier = 0
            if (!$state.show_true_position) {
                const explore_angle = $state.sensor_explore_angle;
                if (explore_angle > -1) {
                    angle_modifier = explore_angle
                } else {
                    angle_modifier = $state.current_seed || Math.random() * 2 * Math.PI;
                }
            }
            return Array.from($state.robot_readings).flatMap((r, i) => {
                let angle = heading + (i * Math.PI * 2) / n_sensors;
                angle += angle_modifier;
                const from = [0, 0, i]
                const to = [r * Math.cos(angle), r * Math.sin(angle), i]
                return [from, to]
            })
           """,
            expression=False,
        ),
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
    js("$state.show_true_position && $state.robot_pose"),
    [
        Plot.line(
            js("$state.true_path"), stroke=Plot.constantly("True Path"), strokeWidth=2
        ),
        sensor_rays,
        Plot.text(
            js("[$state.robot_pose]"),
            text=Plot.constantly(emoji.robot),
            fontSize=30,
            textAnchor="middle",
            dy="-0.35em",
            rotate=js("(-$state.robot_pose[2] + Math.PI/2) * 180 / Math.PI"),
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
    strokeWidth=6,
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
        return $state.possible_paths.map((path, pathIdx) =>
            path.map(([x, y]) => [x, y, pathIdx])
        ).flat()
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


def print_state(w, _):
    """Print current walls and robot path in a format suitable for benchmarking"""
    print("# Benchmark State:")
    print(f"walls = {w.state.walls}")
    print(f"robot_path = {w.state.robot_path}")


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
    [selectable_button, {"onClick": print_state}, "Print State"],
]


def handleSeedIndex(w, e):
    # Called by seed scrubber UI with:
    # w: Plot.js widget
    # e: {index: stripe index, key: current seed}
    # The index indicates which stripe was clicked:
    #   -1: Recycle button was clicked, cycle to next seed
    #    0: First stripe clicked, use first seed
    #   >0: Other stripes clicked, use seed at that position
    # Need to: Get seed from global key based on index, update simulation
    global key
    seed = None
    try:
        if e.index == 0:
            # For first stripe, use first seed from key
            seed = key[0]
        elif e.index == -1:
            # For recycle button, split key into 2 parts and use first seed
            # This cycles through seeds by taking first part
            key = split(key, 2)[0]
            seed = key[0]
        else:
            # For other stripes, split key into e.index parts
            # Use seed at position (index-1) since we're 0-based
            seed = split(key, e.index)[e.index - 1][0]
    except Exception as err:
        # Log any errors that occur during seed selection
        print(f"Error handling seed index: {err}, {e.key}, {e.index}")
    # Update simulation with the selected seed
    update_robot_simulation(w, e, seed=seed)


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
            "True Path": "black",
            "Robot Path": "blue",
        }
    )
    + Plot.colorLegend()
    + Plot.clip()
    + Plot.gridX(interval=1)
)

(
    (
        canvas
        | Plot.js(
            "$state.simulation_time && `${$state.simulation_time?.toFixed(2)} ms`"
        )
    )
    & (
        sliders
        | toolbar
        | true_position_toggle
        | rotating_sensor_rays
        | v.seed_scrubber(handleSeedIndex)
    )
    & {"widths": ["400px", 1]}
    | Plot.initialState(
        create_initial_state(7 + 5 + 14), sync={"current_seed", "selected_tool"}
    )
    | Plot.onChange(
        {
            "robot_path": update_robot_simulation,
            "sensor_noise": update_robot_simulation,
            "motion_noise": update_robot_simulation,
            "heading_noise_scale": update_robot_simulation,
            "n_sensors": update_robot_simulation,
            "walls": update_robot_simulation,
        }
    )
)
