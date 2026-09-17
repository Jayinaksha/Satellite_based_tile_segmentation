import pytest
import tensorflow as tf
from src.models.losses import jaccard_distance, categorical_focal_loss

class TestJaccardDistance:
    def test_perfect_prediction(self):
        y_true = tf.constant([[0, 1, 2]], dtype=tf.int32)
        y_pred = tf.one_hot(y_true, depth=3)
        loss = jaccard_distance(y_true, y_pred, num_classes=3)
        assert loss.numpy() < 0.1

    def test_loss_is_positive(self):
        y_true = tf.constant([[0, 1, 2]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 3, 3))
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        loss = jaccard_distance(y_true, y_pred, num_classes=3)
        assert loss.numpy() >= 0

class TestCategoricalFocalLoss:
    def test_returns_callable(self):
        loss_fn = categorical_focal_loss(gamma=2., alpha=.25, num_classes=6)
        assert callable(loss_fn)

    def test_loss_is_positive(self):
        loss_fn = categorical_focal_loss(gamma=2., alpha=.25, num_classes=3)
        y_true = tf.constant([[0, 1, 2]], dtype=tf.int32)
        y_pred = tf.nn.softmax(tf.random.uniform((1, 3, 3)), axis=-1)
        loss = loss_fn(y_true, y_pred)
        assert tf.reduce_all(loss >= 0)
