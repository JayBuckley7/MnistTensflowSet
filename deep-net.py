import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data;
import os;
import cv2;
import mss
import mss.tools
import numpy as np;
import keras;

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';
SCALAR_RED = (0.0, 0.0, 255.0);
SCALAR_BLUE = (255.0, 0.0, 0.0);


'''
    Class for creating and passing network properties
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

    ##Train(cycles, NetModel);
    Test(cycles, NetModel);

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
        saver = tf.train.Saver();
        tf_log = 'tf.log'
        accuracy = "";
        print("attempting to load first model from storage");
        for epoch in range(numEpochs): #{

            try:#{
                saver.restore(sess,"Model/model.ckpt");
                correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'));
                print("restoring model with ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) * 100,
                      "% accuracy");
                # }
            except Exception as e:#{
                print(str(e));
                print("failed to load  a model from storage... are you sure you had one?");
                # }


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
            saver.save(sess, "Model/model.ckpt");
            print("saving model with ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels})*100,"% accuracy")
    #}
#}



'''
Does the thing
'''
def Test(numEpochs, NetModel:NetworkModel):#{
    #####################
    n = NetModel;
    x = tf.placeholder('float', [None, n.Shape[1]])
    y = tf.placeholder('float')
    pred= GetNNModel(x, n); #one_hot array
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost);
    #######################

    with tf.compat.v1.Session() as sess:
    #{
        print("restoring model");
        saver = tf.train.Saver();
        saver.restore(sess, "Model/model.ckpt");
        for fileName in os.listdir("test_images"):
            print(fileName);
            img = cv2.imread("test_images/"+fileName);
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            gray = cv2.resize(gray, (28, 28));
            gray = gray.astype('float32')
            gray = gray.reshape(1, 784)
            gray /= 255;  ##idk yet

            epic_x = gray;
            epic_y = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0];

            result = sess.run([optimizer, cost], feed_dict={x: epic_x, y: epic_y});
            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));

            tfmax = tf.argmax(pred, 1)
            print("tfMax: ", tfmax);
            prediction = str((tfmax.eval(feed_dict={x: gray})))

            big = cv2.resize(img, (100, 100));
            writeResultOnImage(big, "" + str(prediction))
            print("predicted: "+prediction)
            cv2.imshow("predicted", big)
            cv2.waitKey(0);

#}
#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape
    imageHeight=imageHeight
    imageWidth=imageWidth

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function














if __name__ == "__main__":
    main()