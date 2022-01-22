import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Dropout, ReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.losses import BinaryCrossentropy

WEIGHTS_INITIALIZER = tf.random_normal_initializer(0, 0.02)

def encoder_block(filters, batchnorm=True):
    model = Sequential()

    model.add(
        Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=WEIGHTS_INITIALIZER,
            use_bias=False)
    )

    if batchnorm:
        model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    return model


def decoder_block(filters, dropout=False):
    model = Sequential()
    
    model.add(
        Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=WEIGHTS_INITIALIZER,
            use_bias=False)
    )

    model.add(BatchNormalization())

    if dropout:
        model.add(Dropout(rate=0.5))

    model.add(ReLU())

    return model


class Pix2PixGenerator(Model):
    '''
    Pix2Pix Generator architecture implemented as a Keras model.

    "Image-to-Image Translation with Conditional Adversarial Networks"
    Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
    https://arxiv.org/pdf/1611.07004.pdf
    '''
    
    def __init__(self, reg, initializer=WEIGHTS_INITIALIZER):
        super().__init__()
        
        self.reg = reg
        self.initializer = initializer
        
        self.encoder_blocks = (
            encoder_block(64, batchnorm=False),
            encoder_block(128),
            encoder_block(256),
            encoder_block(512),
            encoder_block(512),
            encoder_block(512),
            encoder_block(512),
            encoder_block(512),
        )

        self.decoder_blocks = (
            decoder_block(512, dropout=True),
            decoder_block(512, dropout=True),
            decoder_block(512, dropout=True),
            decoder_block(512),
            decoder_block(256),
            decoder_block(128),
            decoder_block(64),
        )

        self.last_layer = Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=self.initializer,
            activation='tanh'
        )
        
        self.bce = BinaryCrossentropy(from_logits=True) # use sigmoid
        
    def call(self, X):
        # X shape: (B, 256, 256, 3)
        
        skips = []
        for model in self.encoder_blocks:
            X = model(X)
            skips.append(X)

        skips = reversed(skips[:-1])
        for model, skip in zip(self.decoder_blocks, skips):
            X = model(X)
            X = Concatenate()((X, skip))

        X = self.last_layer(X)
        
        # X shape: (B, 256, 256, 3)
        return X
    
    def compute_loss(self, disc_fake_output, pred_image, target_image, lamda=100):
        labels = tf.ones_like(disc_fake_output)
        loss = self.bce(labels, disc_fake_output)

        if self.reg == 'l1':
            loss += lamda * tf.reduce_mean(tf.abs(target_image - pred_image))
        elif self.reg == 'l2':
            loss += lamda * tf.reduce_mean(tf.math.pow(target_image - pred_image, 2))

        return loss


class Pix2PixDiscriminator(Model):
    '''
    Pix2Pix Discriminator architecture implemented as a Keras model.

    "Image-to-Image Translation with Conditional Adversarial Networks"
    Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
    https://arxiv.org/pdf/1611.07004.pdf
    '''
    
    def __init__(self, initializer=WEIGHTS_INITIALIZER):
        super().__init__()
        
        self.initializer = initializer
        
        self.encoder_block1 = encoder_block(64, batchnorm=False)
        self.encoder_block2 = encoder_block(128)
        self.encoder_block3 = encoder_block(256)
        
        self.zero_pad1 = ZeroPadding2D()
        
        self.conv1 = Conv2D(
            filters=512,
            kernel_size=4,
            strides=1,
            kernel_initializer=self.initializer,
            use_bias=False
        )

        self.batchnorm = BatchNormalization()

        self.lrelu = LeakyReLU(alpha=0.2)

        self.zero_pad2 = ZeroPadding2D()

        self.conv2 = Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            kernel_initializer=self.initializer
        )
        
        self.bce = BinaryCrossentropy(from_logits=True) # use sigmoid
        
    def call(self, inputs):
        input_image, pred_image = inputs
        X = tf.concat((input_image, pred_image), axis=-1)
        # X shape: (B, 256, 256, 2*3)

        X = self.encoder_block1(X)
        X = self.encoder_block2(X)
        X = self.encoder_block3(X)
        X = self.zero_pad1(X)
        X = self.conv1(X)
        X = self.batchnorm(X)
        X = self.lrelu(X)
        X = self.zero_pad2(X)
        X = self.conv2(X)
        
        # X shape: (B, 30, 30, 1)
        return X
        
    def compute_loss(self, disc_real_output, disc_fake_output):
        real_labels = tf.ones_like(disc_real_output)
        real_loss = self.bce(real_labels, disc_real_output)

        fake_labels = tf.zeros_like(disc_fake_output)
        fake_loss = self.bce(fake_labels, disc_fake_output)

        return real_loss + fake_loss


class Pix2Pix(Model):
    '''
    Pix2Pix architecture implemented as a Keras model.

    "Image-to-Image Translation with Conditional Adversarial Networks"
    Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
    https://arxiv.org/pdf/1611.07004.pdf
    '''
    
    def __init__(self, reg='l1'):
        super().__init__()

        self.generator = Pix2PixGenerator(reg)
        self.discriminator = Pix2PixDiscriminator()
        
    def call(self, X):
        return self.generator(X)
    
    def compile(self, generator_opt, discriminator_opt):
        super().compile()
        
        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt
    
    @tf.function
    def train_step(self, data):
        input_image, target_image = data
        
        input_image.set_shape((None, None, None, 3))
        target_image.set_shape((None, None, None, 3))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_real_output = self.discriminator((input_image, target_image), training=True)

            pred_image = self.generator(input_image, training=True)
            disc_fake_output = self.discriminator((input_image, pred_image), training=True)

            gen_loss = self.generator.compute_loss(disc_fake_output, pred_image, target_image)
            disc_loss = self.discriminator.compute_loss(disc_real_output, disc_fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        
        metrics = { 'gen_loss': gen_loss, 'disc_loss': disc_loss }
        return metrics