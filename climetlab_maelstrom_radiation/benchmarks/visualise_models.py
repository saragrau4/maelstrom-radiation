from models import build_cnn, build_fullcnn, build_rnn
from tensorflow.keras.utils import plot_model
from data import load_data


train = load_data(
    mode="train",
    batch_size=256,
    synthetic_data=False,
    cache=False,
    minimal=False,
    tier=1,
)


model = build_cnn(
    train.element_spec[0],
    train.element_spec[1],
)

plot_model(model, to_file="min_cnn.png")

model = build_rnn(
    train.element_spec[0],
    train.element_spec[1],
)

plot_model(model, to_file="rnn.png")

model = build_fullcnn(
    train.element_spec[0],
    train.element_spec[1],
)

plot_model(model, to_file="cnn.png")


model = build_fullcnn(train.element_spec[0], train.element_spec[1], attention=True)

plot_model(model, to_file="cnn_att.png")
