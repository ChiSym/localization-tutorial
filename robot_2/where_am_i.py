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
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
import jax

import robot_2.visualization as v
import robot_2.robot as robot 

key = jax.random.PRNGKey(0)

def convert_walls_to_jax(walls_list: List[List[float]]) -> jnp.ndarray:
    """Convert wall vertices from UI format to JAX array of wall segments"""
    if not walls_list:
        return jnp.array([]).reshape((0, 2, 2))
    
    points = jnp.array(walls_list, dtype=jnp.float32)
    p1 = points[:-1]
    p2 = points[1:]
    
    segments = jnp.stack([
        p1[:, :2],
        p2[:, :2]
    ], axis=1)
    
    valid_mask = p1[:, 2] == p2[:, 2]
    return segments * valid_mask[:, None, None]

def debug_reality(widget, e, subkey=None):
    """Handle updates to robot simulation"""
    if not widget.state.robot_path:
        return
    
    current_key = subkey if subkey is not None else key
    
        
    settings = robot.RobotSettings(
        p_noise=widget.state.motion_noise,
        hd_noise=widget.state.motion_noise,
        sensor_noise=widget.state.sensor_noise,
    )
    
    path = jnp.array(widget.state.robot_path, dtype=jnp.float32)
    walls = convert_walls_to_jax(widget.state.walls)
    n_sensors = int(widget.state.n_sensors)
    
    start_pose = robot.Pose(path[0, :2], 0.0)
    controls = robot.path_to_controls(path)
    
    key_true, key_possible = jax.random.split(current_key)
    
    (final_pose, _), (poses, readings) = robot.simulate_robot_path(
        start_pose, n_sensors, controls, walls, settings, key_true
    )
    
    true_path = jnp.concatenate([start_pose.p[None, :], jax.vmap(lambda p: p.p)(poses)])
    possible_paths = robot.sample_possible_paths(
        key_possible, 20, n_sensors, path, walls, settings
    )
    
    widget.state.update({
        "robot_pose": {
            "x": float(final_pose.p[0]),
            "y": float(final_pose.p[1]),
            "heading": float(final_pose.hd)
        },
        "possible_paths": possible_paths,
        "sensor_readings": readings[-1] if len(readings) > 0 else [],
        "true_path": [[float(x), float(y)] for x, y in true_path],
        "show_debug": True,
        "current_key": current_key[0]  # Send current key to frontend
    })


# Create the visualization
canvas = v.create_robot_canvas(Plot.js("""({points, simplify}) => {
        mode = $state.selected_tool 
        if (mode === 'walls') {
            $state.update(['walls', 'concat', simplify(0.25)])
        }
        if (mode === 'path') {
            $state.robot_path = simplify(0.25)
        }
    }"""))
sliders = v.create_sliders()
toolbar = v.create_toolbar()
reality_toggle = v.create_reality_toggle()
    
    
key_refresh = (
    [Plot.js("""
             ({children}) => {
                 const [inside, setInside] = React.useState(false)
                 const [waiting, setWaiting] = React.useState(false)
                 const [paused, setPaused] = React.useState(false)
                 
                 const text = paused 
                     ? 'Click to Start'
                     : inside 
                         ? 'Click to Pause'
                         : 'Fresh Keys'
                 
                 const onMouseMove = React.useCallback(async (e) => {
                         if (paused || waiting) return null;
                         const rect = e.currentTarget.getBoundingClientRect();
                         const x = e.clientX - rect.left;
                         const stripeIndex = Math.floor(x / stripeWidth);
                         setWaiting(true)
                         await %1({key: $state.current_key, index: stripeIndex});
                         setWaiting(false)
                     })
                 
                 const stripeWidth = 4; // Width of each stripe in pixels
                 
                 return html(["div.rounded-lg.p-2.delay-100", {
                     "style": {
                         background: paused
                             ? 'repeating-linear-gradient(90deg,#aaa,#aaa 4px,#ddd 4px,#ddd 8px)'
                             : 'repeating-linear-gradient(90deg,#86efac,#86efac 4px,#bbf7d0 4px,#bbf7d0 8px)',
                         position: 'relative',
                         opacity: waiting ? 0.5 : 1,
                         transition: 'opacity 0.3s ease'
                     },
                     "onMouseEnter": () => !paused && setInside(true),
                     "onMouseLeave": () => setInside(false),
                     "onClick": () => setPaused(!paused),
                     "onMouseMove": onMouseMove
                 }, text])
             }
             """, lambda w, e: debug_reality(w, e, subkey=jax.random.split(jax.random.PRNGKey(e.key), e.index + 1)[e.index])
             
             )]
)

# 

# Combine all components
(
    canvas & 
    (sliders | toolbar | reality_toggle | key_refresh | Plot.js("$state.current_key")) 
    & {"widths": ["400px", 1]}
    | Plot.initialState(v.create_initial_state(key[0]), sync=True)
    | Plot.onChange({
        "robot_path": debug_reality,
        "sensor_noise": debug_reality,
        "motion_noise": debug_reality,
        "n_sensors": debug_reality,
        "walls": debug_reality
    })
)
