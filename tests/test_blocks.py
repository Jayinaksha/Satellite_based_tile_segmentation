import pytest
import tensorflow as tf
from src.models.blocks import conv_block, encoder_block, decoder_block, bottleneck_block

class TestConvBlock:
    def test_output_shape(self):
        x = tf.random.normal((1, 64, 64, 3))
        out = conv_block(x, 32)
        assert out.shape == (1, 64, 64, 32)

    def test_residual_output_shape(self):
        x = tf.random.normal((1, 64, 64, 32))
        out = conv_block(x, 32, residual=True)
        assert out.shape == (1, 64, 64, 64)  # concatenated

class TestEncoderBlock:
    def test_output_shapes(self):
        x = tf.random.normal((1, 64, 64, 3))
        pool, skip = encoder_block(x, 32, residual=False)
        assert pool.shape == (1, 32, 32, 32)
        assert skip.shape == (1, 64, 64, 32)

class TestBottleneckBlock:
    def test_serial_mode(self):
        x = tf.random.normal((1, 16, 16, 256))
        out = bottleneck_block(x, 128, mode='serial')
        assert out.shape[1:3] == (16, 16)

    def test_parallel_mode(self):
        x = tf.random.normal((1, 16, 16, 256))
        out = bottleneck_block(x, 128, mode='parallel')
        assert out.shape[1:3] == (16, 16)
