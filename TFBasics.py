import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():#{
    x1 = tf.constant(5);
    x2 = tf.constant(6);

    result = tf.multiply(x1, x2);

    with tf.compat.v1.Session() as sess:
    #{
        ans = int(sess.run(result));
        print(ans);
    #}
#}
















if __name__ == "__main__":
    main()