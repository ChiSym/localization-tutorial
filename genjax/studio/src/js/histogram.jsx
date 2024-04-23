import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.14/+esm"
import React from "https://cdn.jsdelivr.net/npm/react@18.2.0/+esm";
const { useRef, useEffect } = React
import { createRender, useModelState } from "https://cdn.jsdelivr.net/npm/@anywidget/react@0.0.6/+esm";
import * as DB from "./tdb.js"
 
const histogramPlot = (values) => {
    return Plot.rectY(values, 
        Plot.binX({y: "count"})
        ).plot({y: {grid: true}})
} 

console.log(DB.db)

const render = createRender(() => {
    let [values, setValues] = useModelState("values");
    let elementRef = useRef()

    useEffect(() => {
        if (!elementRef.current) return;
        let plot = histogramPlot(values)
        elementRef.current.appendChild(plot);
        return  () => plot.remove()
    }, [values, elementRef.current])

    return <div ref={elementRef}>
        <button onClick={(e) => setValues([...values, Math.floor(Math.random() * 999)])}>
            Add a random number
        </button>
    </div>
})

export default { render }