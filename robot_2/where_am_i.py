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

from functools import partial
from typing import List, Tuple, Any, Dict

import genjax
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from penzai import pz
from genstudio.plot import js

import robot_2.emoji as emoji
import robot_2.visualization as v

key = PRNGKey(0)

WALL_COLLISION_THRESHOLD = 0.15


@pz.pytree_dataclass
class Pose(genjax.PythonicPytree):
    """Robot pose with position and heading"""

    p: jnp.ndarray  # [x, y]
    hd: jnp.ndarray  # heading in radians
    __hash__ = None

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
        return [*self.p, self.hd]


@pz.pytree_dataclass
class World(genjax.PythonicPytree):
    """The physical environment with walls that robots can collide with"""

    walls: jnp.ndarray  # [N, 2, 2] array of wall segments
    __hash__ = None

    @jax.jit
    def ray_distance(
        self, ray_start: jnp.ndarray, ray_dir: jnp.ndarray, max_dist: float
    ) -> float:
        """Find distance to nearest wall along a ray"""
        if self.walls.shape[0] == 0:  # No walls
            return max_dist + WALL_COLLISION_THRESHOLD

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
        return jnp.where(
            jnp.isinf(min_dist), max_dist + WALL_COLLISION_THRESHOLD, min_dist
        )

    # TODO, turn this into physical_step (with bounce from wall)
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
            (is_path_clear, safe_pos) where safe_pos is either end_pos or the
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
        is_path_clear = wall_dist >= dist

        # Only adjust position if we actually hit a wall
        final_pos = jnp.where(
            is_path_clear, end_pos, start_pos + ray_dir * (wall_dist - collision_radius)
        )

        return is_path_clear, final_pos


@pz.pytree_dataclass
class RobotCapabilities(genjax.PythonicPytree):
    """Physical capabilities and limitations of the robot"""

    p_noise: jnp.ndarray  # Position noise (std dev in meters)
    hd_noise: jnp.ndarray  # Heading noise (std dev in radians)
    sensor_noise: jnp.ndarray  # Sensor noise (std dev in meters)
    n_sensors: int = 8  # Number of distance sensors
    sensor_range: jnp.ndarray = 10.0  # Maximum sensor range in meters


@jax.jit
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


# Create a sample world with walls and obstacles
example_world = World(
    walls=jnp.array(
        [
            # Outer walls forming a room
            [[0.0, 0.0], [10.0, 0.0]],  # Bottom wall
            [[10.0, 0.0], [10.0, 8.0]],  # Right wall
            [[10.0, 8.0], [0.0, 8.0]],  # Top wall
            [[0.0, 8.0], [0.0, 0.0]],  # Left wall
            # Inner obstacles
            [[3.0, 2.0], [3.0, 4.0]],  # Vertical obstacle
            [[6.0, 4.0], [8.0, 4.0]],  # Horizontal obstacle
        ]
    )
)

# Define robot capabilities
example_robot = RobotCapabilities(
    n_sensors=8,
    sensor_range=jnp.array(5.0),
    sensor_noise=jnp.array(0.1),
    p_noise=jnp.array(0.1),  # Position noise
    hd_noise=jnp.array(0.05),  # Heading noise
)

# Create a sample planned path with waypoints
example_planned_path = jnp.array(
    [
        [1.0, 1.0],  # Start at (1,1)
        [4.0, 2.0],  # Move diagonally up and right
        [7.0, 4.0],  # Continue diagonally
        [8.0, 6.0],  # End near top right
    ]
)


# Sample control instructions for testing
example_controls = jnp.array(
    [
        [2.0, 0.0],  # Move forward 2 units
        [1.5, 0.5],  # Move forward 1.5 units while turning right 0.5 rad
        [2.0, -0.3],  # Move forward 2 units while turning left 0.3 rad
        [1.0, 0.8],  # Move forward 1 unit while turning right 0.8 rad
    ]
)
example_control = example_controls[0]  # Just use first control for single control tests

# Create a sample starting pose
example_pose = Pose(
    p=jnp.array([1.0, 1.0]),  # Starting position at (1,1)
    hd=jnp.array(0.0),  # Initial heading of 0 radians (facing right)
)


@genjax.gen
def get_sensor_reading(
    world: World, robot: RobotCapabilities, pose: Pose, angle: jnp.ndarray
) -> jnp.ndarray:
    """Get a single noisy sensor reading at the given angle relative to robot heading"""
    # Get the ray direction vector for this sensor angle
    ray_dir = jnp.array([jnp.cos(pose.hd + angle), jnp.sin(pose.hd + angle)])

    # Get raw distance reading
    distance = world.ray_distance(
        ray_start=pose.p, ray_dir=ray_dir, max_dist=robot.sensor_range
    )

    # Add noise to reading
    noisy_distance = genjax.normal(distance, robot.sensor_noise) @ "sensor_noise"

    return noisy_distance


# Example usage of get_sensor_reading
# Get a reading from sensor pointing 45 degrees (π/4 radians) relative to robot heading
# key = jax.random.PRNGKey(0)
# reading = get_sensor_reading.simulate(key, (sample_world, sample_robot, sample_pose, jnp.array(jnp.pi/4))).get_retval()
# print(f"Sensor reading at 45 degrees: {reading:.2f} units")


@genjax.gen
def get_all_sensor_readings(world: World, robot: RobotCapabilities, pose: Pose):
    """Get noisy sensor readings at evenly spaced angles around the robot"""
    # Calculate angles for max sensors
    MAX_SENSORS = 32
    angle_step = 2 * jnp.pi / robot.n_sensors
    angles = jnp.arange(MAX_SENSORS) * angle_step

    # Create mask based on actual number of sensors
    mask = jnp.arange(MAX_SENSORS) < robot.n_sensors

    # Chain vmap and mask combinators
    masked_readings = get_sensor_reading.mask().vmap(in_axes=(0, None, None, None, 0))
    return masked_readings(mask, world, robot, pose, angles) @ "sensor_readings"


# Example usage
# readings = get_all_sensor_readings.simulate(key, (example_world, example_robot, example_pose)).get_retval()
# print(f"All sensor readings: {readings.value[readings.flag]}")

@genjax.gen
def execute_control(
    world: World,
    robot: RobotCapabilities,
    current_pose: Pose,
    control: jnp.ndarray,
):
    """Execute a control command with noise, stopping if we hit a wall

    Args:
        control: (distance, angle) pair where:
            - distance is how far to move FIRST
            - angle is how much to turn AFTER moving
    """
    dist, angle = control

    uncorrected_pos = current_pose.p + dist * current_pose.dp()
    noisy_pos = (
        genjax.mv_normal_diag(uncorrected_pos, robot.p_noise * jnp.ones(2)) @ "p_noise"
    )
    (no_obstructions, corrected_pos) = world.check_movement(
        start_pos=current_pose.p,
        end_pos=noisy_pos,
    )

    noisy_angle = genjax.normal(current_pose.hd + angle, robot.hd_noise) @ "hd_noise"
    noisy_pose = Pose(p=corrected_pos, hd=noisy_angle)
    readings = get_all_sensor_readings(world, robot, noisy_pose) @ "readings"
    return noisy_pose, (noisy_pose, readings.value * readings.flag)
#
execute_control_sim = jax.jit(execute_control.simulate)

# execute_control_sim(key, (sample_world, sample_robot, sample_pose, sample_control)).get_retval()


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
    _, (path, readings) = execute_control.partial_apply(world, robot).scan()(start, all_controls) @ "trajectory"
    
    return path, readings


# jax.jit(sample_robot_path.simulate)(key, (sample_world, sample_robot, sample_pose, sample_controls))


sample_robot_path_sim = jax.jit(sample_robot_path.simulate)


def sample_possible_paths(
    world: World, robot: RobotCapabilities, planned_path: jnp.ndarray, n_paths: int, key
):
    """Generate n possible paths given the planned path, respecting walls"""
    (start_pose, controls) = path_to_controls(planned_path[:, :2])
    # Create n random keys for parallel simulation
    keys = jax.random.split(key, n_paths)

    # Vectorize simulation across keys
    return jax.vmap(sample_robot_path_sim, in_axes=(0, None))(
        keys,
        (world, robot, start_pose, controls)
    ).get_retval()


# sample_possible_paths(sample_world, sample_robot, sample_planned_path, 20, key)


@jax.jit
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


def update_robot_simulation(widget, e, seed=None):
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
    paths, readings = sample_possible_paths(
        world, robot, path, n_possible + 1, current_key
    )
    
    widget.state.update(
        {
            "possible_paths": paths[1:],
            "true_path": paths[0],
            "robot_readings": readings[0][-1][:robot.n_sensors],
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
    js("""
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
           """, expression=False),
    z="2",
    stroke="red",
    strokeWidth=1,
    marker="circle",
)


rotating_sensor_rays = (
    Plot.line(
        js("""
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
           """, expression=False),
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
    seed = None
    try:
        if e.index == 0:
            seed = key[0]
        elif e.index == -1:
            key = split(key, 2)[0]
            seed = key[0]
        else:
            seed = split(key, e.index)[e.index - 1][0]
    except Exception as err:
        print(f"Error handling seed index: {err}, {e.key}, {e.index}")
    update_robot_simulation(w, e, seed=seed)


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
            "True Path": "black",
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
    | Plot.initialState(create_initial_state(7+5+14), sync={"current_seed"})
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


# %%
# Test path_to_controls with simple examples
# Test with a longer path
test_path = jnp.array(
    [
        [2.0, 5.0],  # Start
        [3.0, 6.0],  # Up and right
        [4.0, 6.0],  # Right
        [5.0, 5.0],  # Down and right
    ]
)

start_pose, controls = path_to_controls(test_path)
print("Test path:")
print(f"Start pose: pos={start_pose.p}, heading={start_pose.hd:.2f} rad")
print("\nControls (distance, angle):")
for i, (dist, angle) in enumerate(controls):
    print(f"  Step {i}: move {dist:.2f} units, turn {angle:.2f} rad")

test_world = World(jnp.array([]))  # Empty world
test_robot = RobotCapabilities(
    p_noise=jnp.array(0.0),  # No noise for testing
    hd_noise=jnp.array(0.0),
    sensor_noise=jnp.array(0.0),
    n_sensors=8,
    sensor_range=jnp.array(10.0),
)

# Execute each control and track state
current_pose = start_pose
print("\nExecuting controls:")
for i, control in enumerate(controls):
    result = execute_control_sim(key, (test_world, test_robot, current_pose, control))
    current_pose = result.get_retval()[0]
    print(f"\nAfter step {i}:")
    print(f"  Position: {current_pose.p}")
    print(f"  Heading: {current_pose.hd:.2f} rad")
    print(f"  Direction vector: {current_pose.dp()}")
