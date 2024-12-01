from genstudio.plot import js
import genstudio.plot as Plot
from robot_2.emoji import robot, pencil
from typing import Dict, List, Union, Any

WALL_WIDTH = 6
PATH_WIDTH = 6

def drawing_system(key, on_complete):
    """Create drawing system for walls and paths"""
    line = Plot.line(
        js(f"$state.{key}"),
        stroke="#ccc", 
        strokeWidth=4,
        strokeDasharray="4"
    )
    
    events = Plot.events({
        "_initialState": Plot.initialState({key: []}),
        "onDrawStart": js(f"""(e) => {{
            $state.{key} = [[e.x, e.y, e.key]];
        }}"""),
        "onDraw": js(f"""(e) => {{
            if ($state.{key}.length > 0) {{
                const last = $state.{key}[$state.{key}.length - 1];
                const dx = e.x - last[0];
                const dy = e.y - last[1];
                $state.update(['{key}', 'append', [e.x, e.y, e.key]]);
            }}
        }}"""),
        "onDrawEnd": js(f"""(e) => {{
            if ($state.{key}.length > 1) {{
                const points = [...$state.{key}, [e.x, e.y, e.key]];
                const ret = {{
                    points: points,
                    simplify: (threshold=0) => {{
                        const result = [points[0]];
                        let lastKept = points[0];
                        
                        for (let i = 1; i < points.length - 1; i++) {{
                            const p = points[i];
                            const dx = p[0] - lastKept[0];
                            const dy = p[1] - lastKept[1];
                            const dist = Math.sqrt(dx*dx + dy*dy);
                            
                            if (dist >= threshold) {{
                                result.push(p);
                                lastKept = p;
                            }}
                        }}
                        
                        result.push(points[points.length - 1]);
                        return result;
                    }}
                }}
                %1(ret)
            }}
            $state.{key} = [];
        }}""", on_complete)
    })
    return line + events

def create_sliders():
    """Create control sliders for robot parameters"""
    return (
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
    
def clear_state(w, _):
    """Reset visualization state"""
    w.state.update(create_initial_state(w.state.current_key) | {"selected_tool": w.state.selected_tool})
    

def create_toolbar():
    """Create toolbar with tool selection buttons"""
    selectable_button = "button.px-3.py-1.rounded.bg-gray-100.hover:bg-gray-300.data-[selected=true]:bg-gray-300"
    
    return Plot.html("Select tool:") | ["div.flex.gap-2",
        [selectable_button, {
            "data-selected": js("$state.selected_tool === 'path'"),
            "onClick": js("() => $state.selected_tool = 'path'")
        }, f"{robot} Path"],
        [selectable_button, {
            "data-selected": js("$state.selected_tool === 'walls'"),
            "onClick": js("() => $state.selected_tool = 'walls'")
        }, f"{pencil} Walls"],
        [selectable_button, {
            "onClick": clear_state
        }, "Clear"]
    ]

def create_reality_toggle():
    """Create toggle for showing true position"""
    return Plot.html("") | ["label.flex.items-center.gap-2.p-2.bg-gray-100.rounded.hover:bg-gray-300", 
        ["input", {
            "type": "checkbox", 
            "checked": js("$state.show_true_position"),
            "onChange": js("(e) => $state.show_true_position = e.target.checked")
        }], "Show true position"]

def create_sensor_rays():
    """Create visualization for sensor rays"""
    return Plot.line(
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

def create_robot_canvas(drawing_system_handler):
    """Create main robot visualization canvas"""
    return (
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
        + drawing_system("current_line", drawing_system_handler)
        
        # Draw planned path
        + Plot.line(
            js("$state.robot_path"),
            stroke=Plot.constantly("Robot Path"),
            strokeWidth=2,
            r=3,
            marker="circle"
        )
        
        # Draw robot and true path when enabled
        + Plot.cond(
            js("$state.show_true_position"), 
            [Plot.text(
                js("[[$state.robot_pose.x, $state.robot_pose.y]]"),
                text=Plot.constantly(robot),
                fontSize=30,
                textAnchor="middle",
                dy="-0.35em",
                rotate=js("(-$state.robot_pose.heading + Math.PI/2) * 180 / Math.PI")), 
                Plot.line(
                    js("$state.true_path"),
                    stroke=Plot.constantly("True Path"),
                    strokeWidth=2
                ),
                create_sensor_rays()
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

def create_initial_state(key) -> Dict[str, Any]:
    """Create initial state for visualization"""
    return {
        "walls": [
            # Frame around domain (timestamp 0)
            [0, 0, 0], [10, 0, 0],  # Bottom
            [10, 0, 0], [10, 10, 0],  # Right
            [10, 10, 0], [0, 10, 0],  # Top
            [0, 10, 0], [0, 0, 0],  # Left
        ],
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
        "show_true_position": False, 
        "current_line": [],
        "current_key": key
    }

