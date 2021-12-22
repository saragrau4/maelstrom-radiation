import tensorflow as tf

def get_synth_data():
    """Creates a set of synthetic random data.
    Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
    tensor
    dtype: Data type for features/images.
    Returns:
    A tuple of tensors representing the inputs and labels.
    """
    # Synthetic input should be within [0, 255].
    input_shapes = {'sca_inputs':(17,),
                    'col_inputs':(137,27),
                    'hl_inputs':(138,2),
                    'inter_inputs':(136,1),
                    'pressure_hl':(138,1)}
    
    inputs = {}
    for key in input_shapes:
        if key == 'pressure_hl':
            inputs[key] = tf.reshape(tf.range(138),(138,1), name = key)
        else:
            inputs[key] = tf.random.truncated_normal(input_shapes[key],
                                                     dtype=tf.float32,
                                                     mean=5,
                                                     stddev=1,
                                                     name=key)
    output_shapes = {'sw':(3,),
                     'hr_sw':(137,1),
                     }
    outputs = {}
    for key in output_shapes:
        outputs[key] = tf.random.truncated_normal(output_shapes[key],
                                                 dtype=tf.float32,
                                                 mean=5,
                                                 stddev=1,
                                                 name=key)
    return inputs, outputs


def get_synth_dataset(batch_size,dataset_size):
    # Large train = 11660 * 256
    # Large val = 795 * 256
    # Sample train & val = 265 * 256

    inputs, outputs = get_synth_data()
    data = tf.data.Dataset.from_tensors((inputs, outputs)).repeat(dataset_size)
    
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data 
   
