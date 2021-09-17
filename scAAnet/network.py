import numpy as np
import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2

from .loss import poisson_loss, NB_loss, ZINB_loss, ZIPoisson_loss

ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)), name="mean")
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

class DispLayer(tf.keras.layers.Layer):
    def __init__(self, units=2000):
        super(DispLayer, self).__init__()
        self.w = tf.Variable(initial_value=tf.random.normal([units]), trainable=True, name='dispersion')

    def call(self, inputs):
        return DispAct(self.w)

class ZFixedLayer(tf.keras.layers.Layer):
    def __init__(self, dim_latent_space):
        super(ZFixedLayer, self).__init__(name='z_fixed')
        self.w = tf.Variable(initial_value=tf.convert_to_tensor(create_z_fixed(dim_latent_space), tf.keras.backend.floatx()), 
                             trainable=True, name='z_fixed')

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

def create_z_fixed(dim_latent_space):
    """
    Creates Coordinates of the Simplex spanned by the Archetypes.
    The simplex will have its centroid at 0.
    The sum of the vertices will be zero.
    The distance of each vertex from the origin will be 1.
    The length of each edge will be constant.
    The dot product of the vectors defining any two vertices will be - 1 / M.
    This also means the angle subtended by the vectors from the origin
    to any two distinct vertices will be arccos ( - 1 / M ).
    :param dim_latent_space:
    :return:
    """

    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2

        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
    return z_fixed

class Autoencoder():   
    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=(64, 32, 64),
                 dispersion = 'gene',
                 l2_coef=0.,
                 l1_coef=0.,
                 l2_enc_coef=0.,
                 l1_enc_coef=0.,
                 ridge=0.,
                 hidden_dropout=0.,
                 input_dropout=0.,
                 batchnorm=True,
                 activation='relu',
                 init='glorot_normal',
                 file_path=None,
                 debug=False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dispersion = dispersion
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.loss = None
        self.file_path = file_path
        self.extra_models = {}
        self.model = None
        self.encoder = None
        self.decoder = None
        self.input_layer = None
        self.sf_layer = None
        self.debug = debug

        if self.output_size is None:
            self.output_size = input_size
            
        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)

    def build_enc(self):

        self.input_layer = Input(shape=(self.input_size,), name='nor_count')
        self.sf_layer = Input(shape=(1,), name='lib_size')
        last_hidden = self.input_layer
        
        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)
            
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            self.center_idx = center_idx
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
                self.num_at = hid_size
                self.dim_latent_space = self.num_at - 1
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                #layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'
                break

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0. and stage in ('center', 'encoder'):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0. and stage in ('center', 'encoder'):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef
                
            if stage == 'center': # yuge
                fc_a = Dense(hid_size, activation='softmax', kernel_initializer=self.init,
                                kernel_regularizer=l1_l2(l1, l2),
                                name=layer_name)(last_hidden)
                fc_b_t = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                kernel_regularizer=l1_l2(l1, l2),
                                name="%s_b_t"%layer_name)(last_hidden)
                # Add archetype regularization loss
                #z_fixed = tf.eye(self.num_at)
                #self.z_fixed = create_z_fixed(self.dim_latent_space)
                #mu_t = tf.matmul(fc_a, self.z_fixed)
                
                mu_t = ZFixedLayer(self.dim_latent_space)(fc_a)
                fc_b = tf.nn.softmax(tf.transpose(fc_b_t), 1)
                self.z_predicted = tf.matmul(fc_b, mu_t)
                
                #self.arch_loss = tf.math.reduce_mean(tf.math.square(self.z_fixed - z_predicted))
                #last_hidden = fc_a
                last_hidden = mu_t
            else:
                last_hidden = Dense(hid_size, activation = tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name=layer_name)(last_hidden)
                if hid_drop > 0.0:
                    last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)
        
        self.encoder_output = last_hidden
        
    def build_dec(self):
        
        last_hidden = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec0')(self.encoder_output)
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec%s"%(i+1))(last_hidden)
        
        self.decoder_output = last_hidden
        self.dec_layer_num = len(self.hidden_size) - self.center_idx - 1
        self.build_output()

    def build_output(self):

        self.loss = tf.keras.losses.MeanSquaredError()
        mean = Dense(self.output_size, kernel_initializer=self.init, activation="softmax",
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='scaled_mean')(self.decoder_output)
        #output = ColwiseMultLayer([mean, self.ls_layer])

        # keep unscaled output as an extra model
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=mean)  # yuge
        self.arch_loss = tf.math.reduce_mean(tf.math.square(self.model.get_layer('z_fixed').get_weights()[0] - self.z_predicted))
        self.model.add_loss(self.arch_loss)

        self.encoder = self.get_encoder()
        
    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            with open(os.path.join(self.file_path, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)

    def get_encoder(self, activation=False):
   
        return Model(inputs=self.model.input,
                    outputs=self.model.get_layer('center').output)
    
    def get_decoder(self):
        
        # Extract decoder fitted weights
        restored_w = []
        for i in range(self.dec_layer_num):
            restored_w.extend(self.model.get_layer('dec%s'%i).get_weights())
        restored_w.extend(self.model.get_layer('scaled_mean').get_weights())
        
        # Construct decoder
        dec_input_layer = Input(shape=(self.dim_latent_space,), name='latent_space')
        last_hidden = dec_input_layer
        for i, hid_size in enumerate(self.hidden_size[(self.center_idx+1):]):
            last_hidden = Dense(hid_size, activation=tf.nn.leaky_relu,
                                kernel_initializer=self.init,
                                kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                name="dec%s"%i)(last_hidden)
            
        mean = Dense(self.output_size, kernel_initializer=self.init, activation="softmax",
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='scaled_mean')(last_hidden)
        
        dec_model = Model(inputs=dec_input_layer, outputs=mean)
        dec_model.set_weights(restored_w)
        
        return dec_model

    def predict(self, nor_count, lib_size, return_info = False):
        
        print('scAAnet: Calculating reconstructions...')
        preds = self.model.predict({'nor_count': nor_count,
                                    'lib_size': lib_size})
        if isinstance(preds, list):
            recon = preds[0]
        else:
            recon = preds

        print('scAAnet: Calculating low dimensional representations...')
        usage = self.encoder.predict({'nor_count': nor_count,
                                      'lib_size': lib_size})
        
        print('scAAnet: Calculating spectra in the original space')
        self.decoder = self.get_decoder()
        spectra = self.decoder.predict(self.model.get_layer('z_fixed').get_weights()[0])

        return {'recon': recon, 'usage': usage, 'spectra': spectra}
    
class PoissonAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(self.output_size, activation="softmax", kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='scaled_mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        self.loss = poisson_loss

        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)
        self.arch_loss = tf.math.reduce_mean(tf.math.square(self.model.get_layer('z_fixed').get_weights()[0] - self.z_predicted))
        self.model.add_loss(self.arch_loss)

        self.encoder = self.get_encoder()
        
class ZIPoissonAutoencoder(Autoencoder):
    
    def build_dec(self):
        
        # pi
        last_hidden_pi = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec_pi0')(Concatenate()([self.encoder_output, self.sf_layer]))
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_pi = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec_pi%s"%(i+1))(last_hidden_pi)
        # Scaled mean
        last_hidden_mean = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec0')(self.encoder_output)
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_mean = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec%s"%(i+1))(last_hidden_mean)
        
        self.decoder_output_mean = last_hidden_mean
        self.decoder_output_pi = last_hidden_pi
        self.dec_layer_num = len(self.hidden_size) - self.center_idx - 1
        self.build_output()

    def build_output(self):

        pi = Dense(self.output_size, activation=None, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.decoder_output_pi)
        mean = Dense(self.output_size, activation="softmax", kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='scaled_mean')(self.decoder_output_mean)
        output = ColwiseMultLayer([mean, self.sf_layer])
        
        self.loss = ZIPoisson_loss

        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        
        self.model = Model(inputs=[self.input_layer, self.sf_layer], 
                           outputs=[output, pi])
        self.arch_loss = tf.math.reduce_mean(tf.math.square(self.model.get_layer('z_fixed').get_weights()[0] - self.z_predicted))
        self.model.add_loss(self.arch_loss)

        self.encoder = self.get_encoder()

    def predict(self, nor_count, lib_size, return_info=False):
        
        preds = super().predict(nor_count, lib_size)

        if return_info:
            _, pi = self.model.predict({'nor_count': nor_count,'lib_size': lib_size})
            preds['pi'] = pi
        
        return preds
        
class NBAutoencoder(Autoencoder):
    
    def build_dec(self):
        
        # Dispersion
        last_hidden_disp = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec_disp0')(Concatenate()([self.encoder_output, self.sf_layer]))
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_disp = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec_disp%s"%(i+1))(last_hidden_disp)
        # Scaled mean
        last_hidden_mean = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec0')(self.encoder_output)
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_mean = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec%s"%(i+1))(last_hidden_mean)
        
        self.decoder_output_mean = last_hidden_mean
        self.decoder_output_disp = last_hidden_disp
        self.dec_layer_num = len(self.hidden_size) - self.center_idx - 1
        self.build_output()

    def build_output(self):
        
        if self.dispersion == 'gene-cell':
            disp = Dense(self.output_size, activation=DispAct,
                               kernel_initializer=self.init,
                               kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                               name='dispersion')(self.decoder_output_disp)
        else:
            disp = DispLayer(self.output_size)(self.decoder_output_disp)
            
        mean = Dense(self.output_size, activation="softmax", kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='scaled_mean')(self.decoder_output_mean)
        output = ColwiseMultLayer([mean, self.sf_layer])
        
        self.loss = NB_loss
        
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        
        self.model = Model(inputs=[self.input_layer, self.sf_layer], 
                           outputs=[output, disp])
        self.arch_loss = tf.math.reduce_mean(tf.math.square(self.model.get_layer('z_fixed').get_weights()[0] - self.z_predicted))
        self.model.add_loss(self.arch_loss)

        self.encoder = self.get_encoder()

    def predict(self, nor_count, lib_size, return_info=False):
        
        preds = super().predict(nor_count, lib_size)

        if return_info:
            _, disp = self.model.predict({'nor_count': nor_count,'lib_size': lib_size},
                                          batch_size=nor_count.shape[0])
            preds['disp'] = disp
        
        return preds
        
class ZINBAutoencoder(Autoencoder):
    
    def build_dec(self):
        
        # Dispersion/pi
        last_hidden_disp_pi = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec_disp_pi0')(Concatenate()([self.encoder_output, self.sf_layer]))
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_disp_pi = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec_disp_pi%s"%(i+1))(last_hidden_disp_pi)
        # Scaled mean
        last_hidden_mean = Dense(self.hidden_size[(self.center_idx+1)], 
                            kernel_initializer=self.init, 
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                            name='dec0')(self.encoder_output)
        
        if len(self.hidden_size) > (self.center_idx+2):
            for i, hid_size in enumerate(self.hidden_size[(self.center_idx+2):]):
                last_hidden_mean = Dense(hid_size, activation=tf.nn.leaky_relu,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                                    name="dec%s"%(i+1))(last_hidden_mean)
        
        self.decoder_output_mean = last_hidden_mean
        self.decoder_output_disp_pi = last_hidden_disp_pi
        self.dec_layer_num = len(self.hidden_size) - self.center_idx - 1
        self.build_output()

    def build_output(self):
        
        if self.dispersion == 'gene-cell':
            disp = Dense(self.output_size, activation=DispAct,
                               kernel_initializer=self.init,
                               kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                               name='dispersion')(self.decoder_output_disp_pi)
        else:
            disp = DispLayer(self.input_size)(self.decoder_output_disp_pi)
        
        pi = Dense(self.output_size, activation=None, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.decoder_output_disp_pi)
        
        mean = Dense(self.output_size, activation='softmax', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='scaled_mean')(self.decoder_output_mean)      
        output = ColwiseMultLayer([mean, self.sf_layer])

        self.loss = ZINB_loss

        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], 
                           outputs=[output, disp, pi])
        self.arch_loss = tf.math.reduce_mean(tf.math.square(self.model.get_layer('z_fixed').get_weights()[0] - self.z_predicted))
        self.model.add_loss(self.arch_loss)

        self.encoder = self.get_encoder()

    def predict(self, nor_count, lib_size, return_info=False):

        # warning! this may overwrite adata.X
        preds = super().predict(nor_count, lib_size)
        
        if return_info:
            _, disp, pi = self.model.predict({'nor_count': nor_count, 'lib_size': lib_size}, 
                                       batch_size=nor_count.shape[0])
            preds['disp'] = disp
            preds['pi'] = pi
        
        return preds
        
AE_types = {'normal': Autoencoder, 'poisson': PoissonAutoencoder,
            'zipoisson': ZIPoissonAutoencoder,
            'nb': NBAutoencoder, 'zinb': ZINBAutoencoder}