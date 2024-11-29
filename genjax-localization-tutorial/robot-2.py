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
# # Robot Localization: Why is it Hard?
#
# Imagine you have a robot in a building and want it to know where it is. You give it:
# 1. A map of walls and obstacles
# 2. Wheels that can move it around
# 3. Distance sensors that measure how far away walls are
#
# Seems simple, right? Just track the wheel movements and use sensors to confirm position.
#
# ## The Problem
#
# In the real world:
# - Wheels slip and drift
# - Sensors are noisy
# - Small errors compound over time
#
# Try it yourself:
# 1. Add some walls by clicking
# 2. Move the robot around
# 3. Adjust the noise levels to see how they affect:
#    - Sensor readings (red lines)
#    - Motion uncertainty (blue cloud)
#
# Notice how quickly uncertainty grows when noise is present!

# %%
# pyright: reportUnusedExpression=false
# pyright: reportUnknownMemberType=false

import genstudio.plot as Plot
from genstudio.plot import js
import numpy as np
import jax.numpy as jnp
from typing import TypedDict, List, Tuple, Any 

_gensym_counter = 0

WALL_WIDTH=6
PATH_WIDTH=6

def gensym(prefix: str = "g") -> str:
    """Generate a unique symbol with an optional prefix, similar to Clojure's gensym.
    
    Args:
        prefix: Optional string prefix for the generated symbol. Defaults to "g".
        
    Returns:
        A unique string combining the prefix and a counter.
    """
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
            label="Sensor Noise"
        )
     & Plot.Slider(
            "motion_noise",
            range=[0, 1],
            step=0.05,
            label="Motion Noise"
        )
)

initial_state = Plot.initialState({
        "walls": [],
        "robot_pose": {"x": 0.5, "y": 0.5, "heading": 0},
        "sensor_noise": 0.1,
        "motion_noise": 0.1,
        "show_sensors": True,
        "selected_tool": "walls",
        "robot_path": []
    })

canvas = (
            # Draw completed walls
            Plot.line(
                js("$state.walls"),
                stroke=Plot.constantly("Walls"),
                strokeWidth=WALL_WIDTH,
                z="2", 
                render=Plot.renderChildEvents({"onClick": js("""(e) => {
                    const z = $state.walls[e.index][2]
                    $state.walls = $state.walls.filter(([x, y, z]) => z === e.index)
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
            # Draw robot path
            + Plot.line(
                js("$state.robot_path"),
                stroke=Plot.constantly("Robot Path"),
                strokeWidth=PATH_WIDTH
            )
            # Draw robot
            + Plot.dot(
                js("[[$state.robot_pose.x, $state.robot_pose.y]]"),
                r=10,
                fill=Plot.constantly("Robot"),
                title="Robot"
            )
            + Plot.domain([0, 10], [0, 10])
            + Plot.grid()
            + Plot.aspectRatio(1)
            + Plot.colorMap({
                "Walls": "#666",
                "Drawing": "#999",
                "Robot Path": "green",
                "Robot": "blue"
            })
            + Plot.colorLegend()
        )

toolbar = Plot.html("Select tool:") | ["div", {"class": "flex gap-2 h-10"},
            ["button", {
                "class": js("$state.selected_tool === 'walls' ? 'px-3 py-1 rounded bg-gray-400 hover:bg-gray-500 active:bg-gray-600' : 'px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 active:bg-gray-400'"),
                "onClick": js("() => $state.selected_tool = 'walls'")
            }, "Draw Walls"],
            ["button", {
                "class": js("$state.selected_tool === 'path' ? 'px-3 py-1 rounded bg-gray-400 hover:bg-gray-500 active:bg-gray-600' : 'px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 active:bg-gray-400'"),
                "onClick": js("() => $state.selected_tool = 'path'")
            }, "Draw Robot Path"]
        ]

instructions = Plot.md("""
1. Draw walls
2. Draw a robot path
3. Adjust noise levels to see how they affect:
   - Sensor readings
   - Motion uncertainty
        """)

canvas & (toolbar | instructions) | initial_state