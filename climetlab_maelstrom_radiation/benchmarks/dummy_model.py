from tensorflow import keras

class DummyModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        self.compiled_metrics.update_state(y, y)
        return {m.name: 0.0 for m in self.metrics}

    def test_step(self, data):
        x, y = data
        self.compiled_metrics.update_state(y, y)
        return {m.name: 0.0 for m in self.metrics}
        
