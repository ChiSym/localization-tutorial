from genstudio.plot import js
import genstudio.plot as Plot
import robot_2.emoji as emoji


def drawing_system(key, on_complete):
    """Create drawing system for walls and paths"""
    line = Plot.line(
        js(f"$state.{key}"), stroke="#ccc", strokeWidth=4, strokeDasharray="4"
    )

    events = Plot.events(
        {
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
            "onDrawEnd": js(
                f"""(e) => {{
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
        }}""",
                on_complete,
            ),
        }
    )
    return line + events


def seed_scrubber(handle_seed_index):
    """Create a scrubber UI component for exploring different random seeds.

    The component shows a striped bar that can be clicked to pause/resume and
    scrubbed to explore different seeds. A recycle button allows cycling through seeds.

    Args:
        handle_seed_index: Callback function that takes a dict with 'key' (current seed)
            and 'index' (stripe index or -1 for cycle) and handles seed changes.

    Returns:
        A Plot.js component containing the scrubber UI.
    """
    return [
        Plot.js(
            """
                ({children}) => {
                    const [inside, setInside] = React.useState(false)
                    const [waiting, setWaiting] = React.useState(false)
                    const [paused, setPaused] = React.useState(false)

                    const text = paused
                        ? 'Click to Start'
                        : inside
                            ? 'Click to Pause'
                            : 'Explore Seeds'

                    const onMouseMove = React.useCallback(async (e) => {
                            if (paused || waiting) return null;
                            const rect = e.currentTarget.getBoundingClientRect();
                            const x = e.clientX - rect.left;
                            const stripeIndex = Math.max(0, Math.floor(x / stripeWidth));
                            setWaiting(true)
                            await %1({key: $state.current_seed, index: stripeIndex});
                            setWaiting(false)
                        })
                    const stripeWidth = 4; // Width of each stripe in pixels

                    return html(["div.flex.flex-col.gap-1", [
                        ["div.flex.flex-row.gap-1",
                        ["div.text-md.flex.gap-2.p-2.border.hover:border-gray-400.cursor-pointer.font-mono.text-center.w-[140px]", {
                            "onClick": () => {
                                navigator.clipboard.writeText($state.current_seed.toString());
                            },
                            "style": {
                                cursor: "pointer"
                            }
                        }, $state.current_seed, ["div.text-gray-500.ml-auto", "copy"]],[
                            ["div.rounded-lg.p-2.delay-100.flex-grow", {
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
                            }, text],
                            ["button.rounded-lg.p-1.text-xl.hover:bg-green-100", {
                                "onClick": async () => {
                                    setWaiting(true);
                                    await %1({key: $state.current_seed, index: -1}); // Special index to indicate cycle
                                    setWaiting(false);
                                },
                                "style": {
                                    opacity: waiting ? 0.5 : 1,
                                    transition: 'opacity 0.3s ease'
                                }
                            }, %2]
                        ]]
                    ]])
                }
                """,
            handle_seed_index,
            emoji.recycle,
        )
    ]
