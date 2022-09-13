import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber()


def top_scaledflux_mse(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return mse(y_true_tmp, y_pred_tmp)


def top_scaledflux_mae(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return mae(y_true_tmp, y_pred_tmp)


def top_scaledflux_huber(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return huber(y_true_tmp, y_pred_tmp)
