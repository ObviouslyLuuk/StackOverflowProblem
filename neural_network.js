
export function seeded_rand(seed=0) {
    /** Inspired by: https://gist.github.com/blixt/f17b47c62508be59987b */
    let rand = seed * 16807 % 0xFFFF
    return rand / 0xFFFF - .5   // range between -0.5 and 0.5
}

export function rand_matrix(d1, d2, seed=0) {
    let matrix = []
    let rand_counter = 0
    for (let i = 0; i < d1; i++) {
        if (d2 == 1) {
            matrix.push(seeded_rand(seed + rand_counter))
            rand_counter += 1
        }
        else {
            matrix.push([])
            for (let j = 0; j < d2; j++) {
                matrix[i].push(seeded_rand(seed + rand_counter))
                rand_counter += 1
            }            
        }
    }
    return matrix
}

function sigmoid(z) {
    return 1/(1 + Math.exp(-z))
}

function sigmoid_prime(z) {
    return sigmoid(z) * (1 - sigmoid(z))
}

function get_max_value_index(list) {
    /** Returns the index of the max value */
    let max = 0
    let max_index = 0
    for (let index in list) {
        if (list[index] > max) {
            max = list[index]
            max_index = index
        }
    }
    return max_index
}

export class Network {
    constructor(data, layers, batch_size=100, seed=0) {
        // Data is a list of objects like {x: input, y: label} with x as a list of input activations and y as a list of possible outcomes where the correct one has 1 as value
        this.data = data
        this.batch_size = batch_size
        this.seed

        // Layers is a list of sizes for each layer like [16,16] excluding the input and output layer as that's included in the data
        layers.push(data[0].y.length) // Add output layer
        layers.unshift(0) // Add empty input layer into layers like [0,16,16,10], this makes things easier later on because the activations and weights layers will be parallel
        this.layers = layers
        this.depth = layers.length        

        // Initialize weights and biases randomly
        this.biases = []
        for (let layer_width of layers) {
            this.biases.push(rand_matrix(layer_width, 1, seed))
        }        
        this.weights = []
        for (let l in layers) {
            let layer_width = layers[l]
            let prev_layer_width = data[0].x.length
            if (l > 1) {prev_layer_width = layers[l - 1];}

            this.weights.push(rand_matrix(layer_width, prev_layer_width, seed))
        }

        this.best_cost = data[0].y.length // This is the maximum cost
        this.best_biases = this.biases
        this.best_weights = this.weights
        this.cost = null
        this.score = null
    }

    SGD(epochs, eta=1, printing=0) {
        // Stochastic Gradient Descent
        let eta_arg = eta

        // Divide data into batches
        if (printing >= 0) console.log("Making Batches")
        let batches = [[]]
        let batch_nr = 0
        for (let index in this.data) {
            if (index % this.batch_size == 0 && index > 0) {
                batches.push([]);
                batch_nr += 1;
            }
            batches[batch_nr].push(this.data[index])
        }

        let not_improved_tracker = 0

        if (printing >= 0) console.log("Starting Training")
        for (let epoch = 0; epoch < epochs; epoch++) {
            if (printing == 0) {
                this.cost = 0
                this.score = 0
                console.log(`======= STARTING EPOCH ${epoch} =======`)
            }
            let epoch_score = 0
            for (let batch_nr in batches) {
                if (printing >= 1) {
                    this.cost = 0
                    this.score = 0
                    console.log(`======= STARTING EPOCH ${epoch}, BATCH ${batch_nr} =======`)
                }

                this.biases = this.best_biases
                this.weights = this.best_weights

                let batch = batches[batch_nr]
                let bias_nudges_list = []
                let weight_nudges_list = []

                // Do backprop on each example in a batch
                for (let i of batch) {
                    let i_nudges = this.backprop(i.x, i.y)
                    bias_nudges_list.push(i_nudges.biases)
                    weight_nudges_list.push(i_nudges.weights)
                }

                // Average the nudges from backprop to get the total gradient and apply it
                for (let l in this.layers) {
                    for (let j = 0; j < this.layers[l]; j++) {                       
                        // Add average of the nudges for this bias
                        let sum = 0
                        for (let bias_nudges of bias_nudges_list) {
                            sum += bias_nudges[l][j]
                        }
                        this.biases[l][j] -= sum * eta / this.batch_size

                        for (let k = 0; k < this.weights[l][j].length; k++) {
                            // Add average of the nudges for this weight
                            let sum = 0
                            for (let weight_nudges of weight_nudges_list) {
                                sum += weight_nudges[l][j][k]
                            }
                            this.weights[l][j][k] -= sum * eta / this.batch_size
                        }
                    }
                }

                // Test if the cost improved
                let new_cost = 0
                for (let i of batch) {
                    new_cost += this.feedforward(i.x, i.y)["cost"]
                }
                new_cost /= this.batch_size

                if (new_cost < this.best_cost) {
                    this.best_cost = new_cost
                    this.best_biases = this.biases
                    this.best_weights = this.weights
                    not_improved_tracker = 0
                    eta = eta_arg
                } else {
                    not_improved_tracker += 1
                }

                // print stuff
                if (printing >= 1) {
                    epoch_score += this.score
                    console.log(`======= Cost: ${new_cost}`)
                    console.log(`======= Best Cost: ${this.best_cost}, ${not_improved_tracker} batches ago`)
                    console.log(`======= Score: ${this.score / batches[batch_nr].length} avg ${epoch_score / (batch_nr*this.batch_size + batches[batch_nr].length)}`)
                    console.log(`======= Eta: ${eta}`)
                }
            }
            if (printing == 0) {
                console.log(`======= Best Cost: ${this.best_cost}, ${not_improved_tracker} batches ago`)
                console.log(`======= Score: ${this.score / this.data.length}`)
            }  
        }
    }

    feedforward(x, y) {
        // Feedforward and get activations
        let activations = [x] // List of each a on each layer (input layer = x)
        let zs = [] // List of each z on each layer (except the input layer of course which will be an empty list)
        let cost = 0
        let score = 0
        for (let l in this.layers) {
            activations.push([])
            zs.push([])

            for (let j = 0; j < this.layers[l]; j++) {
                // Calculate z and append to zs
                let z = this.biases[l][j]
                for (let k = 0; k < this.weights[l][j].length; k++) {
                    let weight = this.weights[l][j][k]
                    z += weight * activations[l - 1][k]
                }
                zs[l].push(z)

                // calculate activation and append to activations
                let a = sigmoid(z)
                activations[l].push(a)

                // Calculate cost if on last layer
                if (l == this.layers.length - 1) {
                    cost += Math.pow(activations[l][j] - parseInt(y[j]), 2)
                }
            }
        }

        // let nothing = false
        // if (nothing) {
        //     // The order of these can be changed
        //     console.log("This will never print") // it doesn't matter what is logged here as long as it's something
        //     console.log([].length) // the length property has to be used here
        // }        

        // Add one to the score if the prediction was correct
        if (get_max_value_index(activations[this.layers.length - 1]) == get_max_value_index(y))
            score += 1

        return {"activations": activations, "zs": zs, "cost": cost, "score": score}
    }

    backprop(x, y) {
        let res = this.feedforward(x, y)
        let activations = res.activations
        let zs = res.zs
        // this.cost += res.cost // Cost is now checked at the end of each batch
        this.score += res.score

        // TODO: This whole next bit could be done prettier with recursion I think
        // Initialize the lists of lists to store the nudges to biases and weights
        let bias_nudges = []
        let weight_nudges = []
        let a_nudges = [] // We only track this to calculate the other nudges
        for (let l in this.layers) {
            bias_nudges.push([])
            weight_nudges.push([])
            a_nudges.push([])
        }

        // Calculate bias and weight nudges for the last layer using their respective derivatives
        let l = this.layers.length - 1
        for (let j = 0; j < this.layers[l]; j++) {
            let d_a = 2*(activations[l][j] - y[j]) // derivative of Cost with respect to the activation (only for the last layer)
            let d_bias = sigmoid_prime(zs[l][j]) * d_a // derivative of Cost with respect to the bias
            a_nudges[l].push(d_a)
            bias_nudges[l].push(d_bias)
            weight_nudges[l].push([])

            for (let k = 0; k < activations[l - 1].length; k++) {
                let d_weight = activations[l - 1][k] * d_bias // derivative of Cost with respect to the weight
                weight_nudges[l][j].push(d_weight)
            }
        }

        // Calculate nudges for the rest of the layers
        for (let l = this.layers.length - 2; l > 0; l--) {
            for (let j = 0; j < this.layers[l]; j++) {
                // d_a is the sum of the derivatives of Cost with respect to all activations on the layer above (in other words, the effect this a has on all the subsequent activations)
                let d_a = 0
                for (let j1 = 0; j1 < this.layers[l + 1]; j1++) { 
                    d_a += this.weights[l + 1][j1][j] * bias_nudges[l + 1][j1]
                }

                let d_bias = sigmoid_prime(zs[l][j]) * d_a // derivative of Cost with respect to the bias
                bias_nudges[l].push(d_bias)
                weight_nudges[l].push([])
    
                for (let k = 0; k < activations[l - 1].length; k++) {
                    let d_weight = activations[l - 1][k] * d_bias // derivative of Cost with respect to the weight
                    weight_nudges[l][j].push(d_weight)
                }                
            }
        }

        return {"biases": bias_nudges, "weights": weight_nudges}
    }
}