import tensorflow as tf

def ResNet(inputs, class_num=4):
    conv_ksize = 16
    conv_strides = 1
    conv_filters = 64
    dropout_rate = 0.5
    pool_size = 2
    pool_strides = 2

    def _residual_block(x, filters, kernel_size, strides, dropout_rate, grow=True):
        if grow:
            short_cut = tf.layers.conv1d(inputs=x, filters=filters, kernel_size=1, padding='SAME', strides=1)
        else:
            short_cut = tf.identity(x)

        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=dropout_rate)
        x = tf.layers.conv1d(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME', strides=strides)

        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=dropout_rate)
        x = tf.layers.conv1d(inputs=x, filters=filters, kernel_size=kernel_size, padding='SAME', strides=strides)
        if grow:
            short_cut = tf.layers.max_pooling1d(short_cut, pool_size=pool_size, strides=pool_strides)
            x = tf.layers.max_pooling1d(x, pool_size=pool_size, strides=pool_strides)
        x = x + short_cut
        return x

    x = tf.layers.conv1d(inputs=inputs, filters=conv_filters, kernel_size=conv_ksize, padding='SAME', strides=conv_strides)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    short_cut = tf.identity(x)
    x = tf.layers.conv1d(inputs=x, filters=conv_filters, kernel_size=conv_ksize, padding='SAME', strides=conv_strides)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv1d(inputs=x, filters=conv_filters, kernel_size=conv_ksize, padding='SAME', strides=conv_strides)

    short_cut = tf.layers.max_pooling1d(short_cut, pool_size=pool_size, strides=pool_strides)
    x = tf.layers.max_pooling1d(x, pool_size=pool_size, strides=pool_strides)
    x = x + short_cut

    k = 1
    for i in range(15):
        if i%4==0 and i>0:
            k+=1
        x = _residual_block(x, filters=conv_filters*k, kernel_size=conv_ksize, strides=conv_strides, dropout_rate=dropout_rate, grow=(i%2==0 and i!=0) )
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,units=class_num)
    
    return x

    
