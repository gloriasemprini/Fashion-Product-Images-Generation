from keras.callbacks import Callback
from utils import ploters


class CustomFitCallback(Callback):
    def __init__(self, generator, epochs_per_fit=25):
        super().__init__()
        self.epochs_per_fit = epochs_per_fit
        self.current_epoch = 0
        self.generator = generator
    
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        if self.current_epoch % self.epochs_per_fit == 0:
            train_x = next(self.generator)
            wathes = train_x[:5]
            ploters.plot_generated_images([wathes], 1, 5)
            result = self.model.predict(wathes)
            ploters.plot_generated_images([result], 1, 5)