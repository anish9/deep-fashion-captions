import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras import layers


def restore_tokenizer(save_name):
    load_tokenizer = tf.keras.models.load_model(save_name)
    vect_layer = load_tokenizer.layers[-1]
    return vect_layer

def get_activation(x,act):
    if act=="relu":
        x = tf.nn.relu(x)
    if act=="gelu":
        x = tf.nn.gelu(x)
    if act=="selu":
        x = tf.nn.selu(x)
    if act=="lrelu":
        x = tf.nn.leaky_relu(x)
    return x



class positional_embed(tf.keras.layers.Layer):
    def __init__(self,embedding_depth,vocab,sequence_length,**kwargs):
        super(positional_embed,self).__init__(**kwargs)
        self.embedding_depth = embedding_depth
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.embed = tf.keras.layers.Embedding(vocab,embedding_depth)
        
        
    def call(self,data):
        batch_dim = tf.shape(data)[0]
        embeds = np.arange(self.embedding_depth)[np.newaxis,:]
        embeds = 1 / np.power(10000, (2 * (embeds//2)) / np.float32(self.embedding_depth))
        location_id = np.arange(self.sequence_length)[:,np.newaxis]
        pos = embeds*location_id
        pos[:,::2] = np.sin(pos[:,::2])
        pos[:,1::2] = np.cos(pos[:,1::2])
        pos = tf.tile(pos[tf.newaxis,:,:],(batch_dim,1,1))
        pos = tf.cast(pos,tf.float32)
        embed = self.embed(data)
        return embed+pos

    def compute_mask(self,data,mask=None):
        data = tf.cast(data,tf.float32)
        return tf.not_equal(0.0,data)
    

    
    def get_config(self):
        config = super(positional_embed, self).get_config()
        config.update({"embedding_depth": self.embedding_depth,
                       "sequence_length":self.sequence_length,
                       "vocab":self.vocab})
        return config
    

    
    
def image_model(image_size,trainable=False):
    base_model = EfficientNetB0(
        input_shape=image_size, include_top=False, weights="imagenet",
    )
    base_model.trainable = trainable
    base_model_out  = base_model.output
    base_model_out =  layers.Reshape((-1,base_model_out.shape[-1]))(base_model_out)
    
    im_model = tf.keras.models.Model(base_model.input,base_model_out)
    return im_model

class encoder_model(tf.keras.models.Model):
    def __init__(self,dense_dim,embed_dim,num_heads,acti_func="relu",**kwargs):
        super(encoder_model,self).__init__(**kwargs)
        self.densedim = dense_dim
        self.embeddim = embed_dim
        self.numheads = num_heads
        
        self.acti_func = acti_func
        self.dense_layer1 = layers.Dense(embed_dim)
        self.dense_layer2 = layers.Dense(embed_dim)
        self.dense_layer3 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()

        self.mha1 = layers.MultiHeadAttention(num_heads,embed_dim)


        
    def call(self,x):
        x1 = self.layernorm1(x) 
        x2 = self.mha1(x1,x1,x1)
        x3 = self.layernorm2(x1+x2)
        x4 = get_activation(self.dense_layer3(x3),self.acti_func)
        return x4
    
    def get_config(self):
        config = super(encoder_model,self).get_config()
        config.update({"dense_dim":self.densedim,
                       "embed_dim":self.embeddim,
                       "num_heads":self.numheads,
                       "acti_func":self.acti_func})
        return config
    
class decoder_model(tf.keras.models.Model):
    def __init__(self, embed_dim, dense_dim, num_heads, acti_func="relu", **kwargs):
        super(decoder_model, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim =dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation=acti_func), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True
        self.acti_func = acti_func
        self.dropout_layer = layers.Dropout(0.1)
        
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.generate_causal_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:,tf.newaxis,:], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        out_1 = self.dropout_layer(out_1)
        
        attention_output_2,att_weights = self.attention_2(query=out_1,value=encoder_outputs,key=encoder_outputs,attention_mask=padding_mask,
                                                         return_attention_scores=True)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)
    
    def generate_causal_mask(self,inputs):
        batch_size,seq_length = tf.shape(inputs)[0],tf.shape(inputs)[1]
        x = tf.range(seq_length)
        y = tf.range(seq_length)[:,tf.newaxis]
        causal_mask = tf.cast(y>=x,dtype="int32")[tf.newaxis,:,:]
        causal_mask = tf.tile(causal_mask,(batch_size,1,1))
        return causal_mask
    
    def get_config(self):
        config = super(decoder_model,self).get_config()
        config.update({"num_heads":self.num_heads,
                      "embed_dim":self.embed_dim,
                      "dense_dim":self.dense_dim,
                      "acti_func":self.acti_func})
        return config