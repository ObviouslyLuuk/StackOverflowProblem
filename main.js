import {Network} from "./neural_network.js"
import {Network as Network_DC} from "./neural_network_with_dead_code.js"

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

function initNet(dead_code=false) {
    let training_data = xor_3
    switch(dead_code) {
        case false:
            var net = new Network(training_data, [10])
            break
        case true:
            var net = new Network_DC(training_data, [10])
    }
    net.SGD(1000, 1, -1)
}

function testNets(round, time_without, time_with) {
    let time_sum = 0
    let time_sum_dead_code = 0
    let i = 0

    function runTests() {
        for (let dead_code of [false, true]) {
            let t0 = performance.now()
            initNet(dead_code)
            let t1 = performance.now()

            switch(dead_code) {
                case false:
                    time_sum += (t1-t0)
                    break
                case true:
                    time_sum_dead_code += (t1-t0)
            }
        }
        round.innerHTML = i + 1
        time_without.innerHTML = time_sum
        time_with.innerHTML = time_sum_dead_code  

        i++
        if (i < 10000) {
            setTimeout(runTests, 1)
        }
    }
    runTests()
}

testNets(round, time_without, time_with)