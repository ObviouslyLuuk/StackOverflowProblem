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

function initNet(dead_code=false, epochs=500) {
    let training_data = xor_3
    switch(dead_code) {
        case false:
            var net = new Network(training_data, [16,16])
            break
        case true:
            var net = new Network_DC(training_data, [16,16])
    }
    net.SGD(epochs, 1, -1)
}

function testNets(round, time_without, time_with, time_difference, time_once, time_once_with, time_once_difference) {

    let element = time_once
    let dead_code = false
    function runTestOnce() {
        let t0 = performance.now()
        initNet(dead_code, 100000)
        let t1 = performance.now()
        element.innerHTML = (t1-t0)
        time_once_difference.innerHTML = (time_once.innerHTML / time_once_with.innerHTML - 1) * 100    

        if (!dead_code) {
            element = time_once_with
            element.innerHTML = "Starting..."
            dead_code = true
            setTimeout(runTestOnce, 1)
        }
    }

    element.innerHTML = "Starting..."
    setTimeout(runTestOnce, 1)
    
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
        time_difference.innerHTML = (time_sum / time_sum_dead_code - 1) * 100

        i++
        if (i < 10000) {
            setTimeout(runTests, 1)
        }
    }
    runTests()
}

testNets(round, time_without, time_with, time_difference, time_once, time_once_with, time_once_difference)