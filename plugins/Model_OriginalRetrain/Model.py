# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, Lambda, MaxPool2D, add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K

from .AutoEncoder import AutoEncoder
from lib.PixelShuffler import PixelShuffler

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024

conv_init = RandomNormal(0, 0.02)

def ground_truth_diff(y_true, y_pred):
    return K.abs(y_pred - y_true)

#from https://github.com/YapengTian/SRCNN-Keras/blob/master/SRCNN_train/SRCNN_train.py
def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

class Model(AutoEncoder):
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)

        encoder = self.encoder
        encoder.trainable = False
        decoder = self.decoder_B
        decoder.trainable = False
        self.autoencoder_B = KerasModel(x, self.refiner(decoder(encoder(x))))

        self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

        encoder.summary()
        decoder.summary()
        self.refiner.summary()
        self.autoencoder_B.summary()

        from keras.utils import plot_model
        plot_model(self.encoder, to_file='_model_encoder.png', show_shapes=True, show_layer_names=True)
        plot_model(self.decoder_B, to_file='_model_decoder_B.png', show_shapes=True, show_layer_names=True)
        plot_model(self.refiner, to_file='_model_refiner.png', show_shapes=True, show_layer_names=True)

    def converter(self, swap):
        autoencoder = self.autoencoder_B
        return lambda img: autoencoder.predict(img)

    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale(512)(x)
        return KerasModel(input_, x)

    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = self.upscale(32)(x)
        x = self.upscale(16)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)

    def Refiner(self):
        
        def srcnn(x, channels):
            x = Conv2D(64, 9, padding='same', activation='relu', init='he_normal')(x)
            x = Conv2D(32, 1, padding='same', activation='relu', init='he_normal')(x)
            x = Conv2D(channels, 5, padding='same', init='he_normal')(x)
            return x
        
        input_ = Input(shape=(256, 256, 3))
        x = input_
        # x = Lambda(lambda image: K.tf.image.rgb_to_hsv(image))(x)
        # Y = Lambda(lambda x : x[:,:,:,:1])(x)
        # Cr = Lambda(lambda x : x[:,:,:,1:2])(x)
        # Cb = Lambda(lambda x : x[:,:,:,2:3])(x)
        # Y = srcnn(Y, 1)
        # x = concatenate([Y, Cr, Cb])
        # x = Lambda(lambda image: K.tf.image.hsv_to_rgb(image))(x) #Fails with 'None value are not supported'
        x = srcnn(x, 3)
        return KerasModel(input_, x)
