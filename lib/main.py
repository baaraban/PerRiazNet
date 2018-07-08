import infrastructure as inf
import functions as func


net = inf.NeuralNet(8, 29)

input_layer = inf.InputLayer(input_size=8)
first_layer = inf.Layer(func.HyperbolicTangent, out_size=18)
second_layer = inf.Layer(func.ReLU, out_size=25)
out_layer = inf.OutputLayer(func.SoftMax, 29, func.CrossEntropy)

input_to_first_connection = inf.Connection(input_layer, 8, 18)
first_layer.add_connection(input_to_first_connection)

first_to_second_connection = inf.Connection(first_layer, 18, 25)
second_layer.add_connection(first_to_second_connection)

second_to_third_connection = inf.Connection(second_layer, 25, 29)
input_to_third_connection = inf.Connection(input_layer, 8, 29)
out_layer.add_connection(second_to_third_connection)
out_layer.add_connection(input_to_third_connection)


net.add_layer(input_layer)
net.add_layer(first_layer)
net.add_layer(second_layer)
net.add_layer(out_layer)
