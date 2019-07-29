import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
    Class for creating and passing network propertys
    because you python programmers are mental
'''
class NetworkModel: #{
  def __init__(self, Layers, NumOutcomes, Shape, BatchSize ):  #{
    self.Layers = Layers;
    self.NumOutcomes = NumOutcomes;
    self.BatchSize = BatchSize;
    self.Shape = Shape
    #}
#}
'''
input->weight->lay1->activation->[layer]->outputlayer

compare to intented output(cost/loss) [cross entropy  >?]
optimize > minimize loss (adamOptimize...sgd...adaGrad)
^^^back prop

feed forward + backprop  is called an epoch or cycle [epic]
'''
def main():#{

    ##define properys for out neural network
    HLNodes = [500,500,500];
    NumOutcomes = 10;
    BatchSize = 100;
    area = 28*28; ## mnist img size
    shape = [None , area]

    NetModel = NetworkModel(HLNodes, NumOutcomes, shape, BatchSize);

    ## height by width of data sets


    cycles = 10;

    Train(cycles, NetModel)

#}


'''
Takes data and NNModel Propertys and  gives back a nice model thing?
@Param data = idk man
@Param NetModel = Gimmie some propertys
'''
def GetNNModel(data, NetModel:NetworkModel):#{
    n = NetModel;
    print("")
    print("=======")
    print("L1: ", n.Layers[0])
    print("L2: ", n.Layers[1])
    print("L3: ", n.Layers[2])
    print("s1: ", n.Shape[0])
    print("s2: ", n.Shape[1])
    print("=======")
    ##TF will generate us some bias to mutate each epoch
    ##define out Hidden layyers
    HLayer1 = {'weights': tf.Variable(tf.random_normal([n.Shape[1], n.Layers[0]])),
               'biases': tf.random_normal([n.Layers[0]])};

    HLayer2 = {'weights': tf.Variable(tf.random_normal([n.Layers[0], n.Layers[1]])),
               'biases': tf.random_normal([n.Layers[1]])};

    HLayer3 = {'weights': tf.Variable(tf.random_normal([n.Layers[1], n.Layers[2]])),
               'biases': tf.random_normal([n.Layers[2]])};
    ##define out output layer
    OutputLayer = {'weights': tf.Variable(tf.random_normal([n.Layers[2], n.NumOutcomes])),'biases': tf.random_normal([n.NumOutcomes])};

    ##make out model -> (Input_Data * weights)+Bias
    layer1 = tf.add(tf.matmul(data,  HLayer1['weights']),HLayer1['biases']);
    ##Activate this layer? [some wierd sigmoid thing.)
    layer1 = tf.nn.relu(layer1);

    layer2 = tf.add(tf.matmul(layer1,  HLayer2['weights']),HLayer2['biases']);
    ##Activate this layer? [some wierd sigmoid thing.)
    layer2 = tf.nn.relu(layer2);

    layer3 = tf.add(tf.matmul(layer2,  HLayer2['weights']),HLayer2['biases']);
    ##Activate this layer? [some wierd sigmoid thing.)
    layer3 = tf.nn.relu(layer3);

    output = tf.matmul(layer3, OutputLayer['weights']) + OutputLayer['biases']
    return output
#}


def neural(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784, 500])),
    'biases':tf.Variable(tf.random_normal([500]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([500, 500])),
    'biases':tf.Variable(tf.random_normal([500]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([500, 500])),
    'biases':tf.Variable(tf.random_normal([500]))}
    output_layer={'weights':tf.Variable(tf.random_normal([500, 10])),
    'biases':tf.Variable(tf.random_normal([10]))}

    l1=tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    li= tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2= tf.nn.relu(l2)
    l3=tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3= tf.nn.relu(l3)
    output= tf.matmul(l3, output_layer['weights'])+ output_layer['biases']
    return output


'''
Does the thing
'''
def Train(numEpochs, NetModel:NetworkModel):#{
    n = NetModel;
    x = tf.placeholder('float', [None, n.Shape[1]])
    y = tf.placeholder('float')

    pred= GetNNModel(x, n); #one_hot array

    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost);

    with tf.compat.v1.Session() as sess:
    #{
        sess.run(tf.initialize_all_variables());
        for epoch in range(numEpochs): #{
            epoch_cost = 0;
            for _ in range(int(mnist.train.num_examples/n.BatchSize)):
            #{
                epic_x,epic_y = mnist.train.next_batch(n.BatchSize);
                _, c = sess.run([optimizer, cost], feed_dict = {x: epic_x, y: epic_y});
                epoch_cost += c;
                print("epoch ", epoch ," / ", numEpochs, " loss: ",epoch_cost)
            #}
            correct = tf.equal(tf.argmax(pred, 1),tf.argmax(y,1));
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'));
            print('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}));
        #}
    #}
#}

















if __name__ == "__main__":
    main()