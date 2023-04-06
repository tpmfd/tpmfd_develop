from tensorlayer.layers import Layer
import numpy as np
import tensorflow as tf

class mycov2D(Layer):
    def __init__(
            self,
            layer,
            topo_data,
            n_batchsize,
            WW,
            bb,
            shape=(1,1,2,1),
            strides=(1, 1, 1, 1),
            padding='SAME',
            act=tf.nn.relu,
            name='mycov2D',

    ):
        Layer.__init__(self,name=name)

        out_temp0=layer.outputs
        size1=out_temp0.shape
        print(out_temp0.shape)
        for i in range(0,n_batchsize):
            out_temp2=out_temp0[i,:,:,:]
            #print('temp2',out_temp2.shape)
            out_temp3=tf.concat([out_temp2,topo_data],2)
            #print('temp3', out_temp3.shape)

            #out_temp4[i,:,:,:].assign(out_temp3)
            if i==0:
                out_temp4=tf.expand_dims(out_temp3,0)
                #print('out_temp4',out_temp4.shape)
            else:
                out_temp5 = tf.expand_dims(out_temp3, 0)
                out_temp4 =tf.concat([out_temp4,out_temp5],0)

        print(out_temp4.shape)
        self.inputs=out_temp4
        print("mycov2D_layer %s: %s" %(self.name,act))
        n_in=int(self.inputs.shape[-1])

        with tf.variable_scope(name) as vs:

            self.outputs=tf.nn.conv2d(self.inputs, WW, strides=strides, padding=padding) + bb
        self.all_layers=list(layer.all_layers)
        self.all_params=list(layer.all_params)
        self.all_drop=dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend([WW,bb])