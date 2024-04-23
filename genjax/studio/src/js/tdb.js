import { defHashMap, defSortedMap} from "https://cdn.jsdelivr.net/npm/@thi.ng/associative@6.3.57/+esm"
import { hash } from "https://cdn.jsdelivr.net/npm/@thi.ng/vectors@7.10.29/+esm"


export const db = {
    currentTime: 0,
    data: defHashMap({}), 
    
    assoc(time, key, value) {
        if (arguments.length === 1) {
            this.currentTime = time;
        } else if (arguments.length === 2) {
            key = arguments[0]; 
            value = arguments[1];
            this.assoc(this.currentTime, key, value);
        } else {
            const values = this.data.get(key) || defSortedMap({});
            values.set(time, value);
            this.data.set(key, values);
        }
        return this;
    },
    
    get(keyOrTime, optKey) {
        if (arguments.length === 1) {
            return this.get(this.currentTime, keyOrTime);
        } else {
            let values = this.data.get(optKey);
            if (!values) return;
            return values.get(values.lte(keyOrTime));
        }
    }
}
