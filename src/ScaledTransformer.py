import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Layer, Dense, Embedding, SpatialDropout1D, Concatenate, Add, LayerNormalization, Dropout
from tensorflow.keras.models import Model

def FTSwish(threshold=-0.2):
    def _FTSwish(x):
        return K.relu(x) * K.sigmoid(x) + threshold
    return Lambda(_FTSwish)

def pos_enc():
    def _pos_enc(x):
        _, max_len, d_emb = K.int_shape(x)
        
        pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
          for pos in range(max_len)
          ])
        
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        x += pos_enc
        return x
    return Lambda(_pos_enc)
    

def create_padding_mask():
    def _create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, :]
    return Lambda(_create_padding_mask)

def create_look_ahead_mask():
    def _create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    return Lambda(_create_look_ahead_mask)

def create_masks():
    def _create_masks(x):
        inp, tar = x
        enc_padding_mask = create_padding_mask()(inp)
        dec_padding_mask = create_padding_mask()(inp)

        look_ahead_mask = create_look_ahead_mask()(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask()(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask
    return Lambda(_create_masks)

def out_pad_mask():
    def _out_pad_mask(x):
        tensor, seq = x
        n = tf.math.divide_no_nan(seq, seq)
        n = K.expand_dims(n, axis=-1)
        mask = tf.tile(n, [1, 1, K.int_shape(tensor)[-1]])
        
        return tensor*mask
    return Lambda(_out_pad_mask)

        
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, depth, N, typ):
        super(SelfAttention, self).__init__()
        
        self.depth = depth
        self.N = N
        self.typ = typ
        
    def build(self, input_shape):
        q,k,v,m = input_shape
        scale = np.power(9*self.N, -1/4) if self.typ=='de' else np.power(self.N, -1/4)*0.67
        
        self.wq = self.add_weight(shape=(q[-1], self.depth),
                                  initializer='glorot_normal', trainable=True, name='q_kernel')
        
        self.wk = self.add_weight(shape=(k[-1], self.depth),
                                  initializer='glorot_normal', trainable=True, name='k_kernel')

        self.wv = self.add_weight(shape=(v[-1], self.depth),
                                  initializer=lambda shape, dtype=None: 
                                  tf.random.uniform(shape,dtype=dtype, minval=-np.sqrt(6 / (v[-1]+self.depth)), 
                                                    maxval=np.sqrt(6 / (v[-1]+self.depth)))*scale, 
                                  trainable=True, name='v_kernel')

        self.built = True
        
    def call(self, x):
        query, key, value, mask = x

        q = tf.matmul(query, self.wq)
        k = tf.matmul(key, self.wk)
        v = tf.matmul(value, self.wv)

        g = tf.linalg.matmul(q, k, transpose_b=True) / K.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            g += (mask * -1e9) 

        g = K.softmax(g, axis=-1)
        att = tf.linalg.matmul(g, v)
        return att
        
def Transformer(num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target, rate):
    inp = Input(shape=(pe_input))
    tar_inp = Input(shape=(pe_target-1))
    tar_real = Input(shape=(pe_target-1))

    #masks
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks()([inp, tar_inp])
    
    Shared_embedding = Embedding(vocab_size, d_model, 
                     embeddings_initializer=tf.keras.initializers.TruncatedNormal(), 
                                 mask_zero=True)

    enc_emb = Shared_embedding(inp)
    enc_emb = Lambda(lambda x: x*np.power(9*num_layers, -1/4))(enc_emb)
    enc_emb = Lambda(lambda x : x*tf.math.sqrt(tf.cast(d_model, tf.float32)))(enc_emb)
    enc_emb = pos_enc()(enc_emb)

    x = Dropout(rate)(enc_emb)

    #encoder
    for i in range(1, num_layers+1):
        #Mha
        heads = []
        for head in range(num_heads):
            att = SelfAttention(int(d_model/num_heads), num_layers, 'en')([x, x, x, enc_padding_mask])
            heads.append(att)
        att = Concatenate()(heads)
        att = Dense(d_model, 
                    kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / (d_model+d_model)), 
                                      maxval=np.sqrt(6 / (d_model+d_model)))*np.power(num_layers, -1/4)*0.67, 
                    name=f'att_dense_{i}_layer')(att)
        
        att = Dropout(rate)(att)
        x = Add()([att, x])
        #FF
        d = Dense(dff, 
                  kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / d_model), 
                                      maxval=np.sqrt(6 / d_model))*np.power(num_layers, -1/4)*0.67,
                  name=f'FF_dff_{i}_layer')(x)
        d = FTSwish()(d)
        d = Dense(d_model, 
                  kernel_initializer=lambda shape, dtype=None: 
                  tf.random.uniform(shape,dtype=dtype, 
                                    minval=-np.sqrt(6 / (d_model+dff)), 
                                    maxval=np.sqrt(6 / (d_model+dff)))*np.power(num_layers, -1/4)*0.67, 
                  name=f'FF_dense_{i}_layer')(d)
        d = Dropout(rate)(d)
        x = Add()([d, x])
    
    enc_out = out_pad_mask()([x, inp])

    
    dec_emb = Shared_embedding(tar_inp)
    dec_emb = Lambda(lambda x: x*np.power(9*num_layers, -1/4))(dec_emb)
    dec_emb = Lambda(lambda x : x*tf.math.sqrt(tf.cast(d_model, tf.float32)))(dec_emb)
    dec_emb = pos_enc()(dec_emb)   
    
    y = Dropout(rate)(dec_emb)
            
    #decoder
    for i in range(1, num_layers+1):
        #Self-decoder Mha
        s_heads = []
        for head in range(num_heads):
            att = SelfAttention(int(d_model/num_heads), num_layers, 'de')([y, y, y, combined_mask])
            s_heads.append(att)
        att = Concatenate()(s_heads)
        att = Dense(d_model, 
                    kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / (d_model+d_model)), 
                                      maxval=np.sqrt(6 / (d_model+d_model)))*np.power(num_layers*9, -1/4), 
                    name=f'self_dec_dense_{i}_layer')(att)
            
        att = Dropout(rate)(att)
        y = Add()([att, y])
        
        #Encoder-decoder Mha
        e_heads = []
        for head in range(num_heads):
            att = SelfAttention(int(d_model/num_heads), num_layers, 'de')([y, enc_out, enc_out, dec_padding_mask])
            e_heads.append(att)
        att = Concatenate()(e_heads)
        att = Dense(d_model, 
                    kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / (d_model+d_model)), 
                                      maxval=np.sqrt(6 / (d_model+d_model)))*np.power(num_layers*9, -1/4), 
                    name=f'enc_dec_dense_{i}_layer')(att)
            
        att = Dropout(rate)(att)
        y = Add()([att, y])
            
        #FF
        d = Dense(dff, 
                  kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / d_model), 
                                      maxval=np.sqrt(6 / d_model))*np.power(num_layers*9, -1/4), 
                  name=f'dec_FF_dff_{i}_layer')(y)
        d = FTSwish()(d)
        d = Dense(d_model, 
                  kernel_initializer=lambda shape, dtype=None: 
                    tf.random.uniform(shape,dtype=dtype, 
                                      minval=-np.sqrt(6 / (d_model+dff)), 
                                      maxval=np.sqrt(6 / (d_model+dff)))*np.power(num_layers*9, -1/4), 
                  name=f'dec_FF_dense_{i}_layer')(d)
        d = Dropout(rate)(d)
        y = Add()([d, y])
                
    #final layer
#     output = Dense(d_model, kernel_initializer='glorot_normal')(y)
#     output = FTSwish()(output)
    output = Dense(vocab_size, activation='softmax', kernel_initializer='glorot_normal', name='final_dense')(y)#output#
#     output = out_pad_mask()([output, tar_real])
    
    return Model(inputs=[inp, tar_inp, tar_real], outputs=output)
    