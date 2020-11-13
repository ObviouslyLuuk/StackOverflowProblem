import {Network} from "./neural_network.js"

// XOR with 3 inputs
const xor_3 = [
    {"x": [0,0,0], "y": [1,0]},
    {"x": [1,0,0], "y": [0,1]},
    {"x": [0,1,0], "y": [0,1]},
    {"x": [1,1,0], "y": [1,0]},
    {"x": [0,0,1], "y": [0,1]},
    {"x": [1,0,1], "y": [1,0]},
    {"x": [0,1,1], "y": [1,0]},
    {"x": [1,1,1], "y": [1,0]},        
]

function initNet() {
    let training_data = xor_3
    let net = new Network(training_data, [10])
    net.SGD(100000, 1, 0)
}

initNet()