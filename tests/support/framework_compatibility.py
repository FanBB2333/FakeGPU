"""Numerical framework probes; every selected probe must actually execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def tensorflow_probe():
    import numpy as np
    import tensorflow as tf

    from fakegpu.tensorflow import simulate

    x = np.array([[1., 2.], [2., -1.], [-1., 1.]], dtype=np.float64)
    expected_weights = np.array([[2.], [-1.]], dtype=np.float64)
    y = x @ expected_weights
    original_device = tf.device
    with simulate(device_count=2):
        with tf.device("/GPU:1"):
            inputs = tf.constant(x)
            weights = tf.Variable([[0.], [0.]], dtype=tf.float64)
            np.testing.assert_allclose(tf.matmul(inputs, expected_weights).numpy(), y)
            assert "CPU:0" in inputs.device

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(inputs @ weights - y))
        gradient = tape.gradient(loss, weights)
        np.testing.assert_allclose(gradient.numpy(), 2 * x.T @ -y / len(x))

        @tf.function(autograph=False)
        def train_step():
            with tf.device("/GPU:1"):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(inputs @ weights - y))
                weights.assign_sub(0.05 * tape.gradient(loss, weights))
                return loss

        reference = np.zeros((2, 1))
        for _ in range(20):
            train_step()
            reference -= 0.05 * (2 * x.T @ (x @ reference - y) / len(x))
        np.testing.assert_allclose(weights.numpy(), reference, rtol=1e-10, atol=1e-10)

        with tf.device("/GPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,), dtype="float64"),
                tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros", dtype="float64"),
            ])
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss="mse")
            model.train_on_batch(x, y)
            expected = -0.05 * (2 * x.T @ -y / len(x))
            np.testing.assert_allclose(model(x).numpy(), x @ expected, rtol=1e-6, atol=1e-6)
            assert "CPU:0" in model(x).device

    assert tf.device is original_device
    return {
        "framework": "tensorflow",
        "version": tf.__version__,
        "execution": "cpu_device_scope_adapter",
        "checks": ["eager_matmul", "autodiff", "graph_training", "keras_training", "cpu_placement"],
    }


def sklearn_cpu_probe():
    import numpy as np
    import sklearn
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.array([[1., 2.], [2., -1.], [-1., 1.], [3., 4.], [-2., -3.]])
    y = x @ np.array([2., -1.]) + 3
    regressor = make_pipeline(StandardScaler(), LinearRegression()).fit(x, y)
    np.testing.assert_allclose(regressor.predict(x), y, atol=1e-10)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=100)).fit(x, x[:, 0] > 0)
    np.testing.assert_array_equal(classifier.predict(x), x[:, 0] > 0)
    return {
        "framework": "scikit-learn",
        "version": sklearn.__version__,
        "execution": "ordinary_cpu",
        "checks": ["preprocessing", "regression", "classification"],
    }


def sklearn_array_api_probe():
    import numpy as np
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    import fakegpu

    fakegpu.init(runtime="fakecuda", device_count=2, profile="a100")
    import torch

    x = np.array([[1., 2.], [2., -1.], [-1., 1.], [3., 4.], [-2., -3.]])
    y = x @ np.array([2., -1.])
    fake_x = torch.asarray(x, device="cuda:1")
    fake_y = torch.asarray(y, device="cuda:1")
    for create in (StandardScaler, lambda: PCA(n_components=1), lambda: Ridge(alpha=.1, solver="svd")):
        reference = create().fit(x, y)
        with sklearn.config_context(array_api_dispatch=True):
            model = create().fit(fake_x, fake_y)
            actual = model.predict(fake_x) if hasattr(model, "predict") else model.transform(fake_x)
        expected = reference.predict(x) if hasattr(reference, "predict") else reference.transform(x)
        assert actual.device == torch.device("cuda:1")
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-10, rtol=1e-10)
    return {
        "framework": "scikit-learn",
        "version": sklearn.__version__,
        "torch_version": torch.__version__,
        "execution": "fakecuda_cpu_backing",
        "checks": ["array_api_scaler", "array_api_pca", "array_api_ridge", "cuda_device_preserved"],
    }


PROBES = {
    "tensorflow": tensorflow_probe,
    "sklearn-cpu": sklearn_cpu_probe,
    "sklearn-array-api": sklearn_array_api_probe,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=PROBES, required=True)
    args = parser.parse_args()
    report = PROBES[args.framework]()
    print(json.dumps({"status": "passed", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
