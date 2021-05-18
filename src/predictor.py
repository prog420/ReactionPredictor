import Algorithmia
import tensorflow as tf
import re
from .ScaledTransformer import Transformer
from .BeamPredictor import Prediction

"""
Example Input:
{
    "reaction": SMILES
}

Expected Output:
{
    "product": SMILES
}
"""

# Print TensorFlow info
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.sysconfig.get_build_info()["cuda_version"])
print(tf.test.is_built_with_gpu_support())
print(tf.config.list_physical_devices("GPU"))
print(tf.test.gpu_device_name())


# Configure Tensorflow to only use up to 30% of the GPU.
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5720)])


class InputObject:
    def __init__(self, input_dict):
        """
        Creates an instance of the InputObject, which checks the format of data and throws exceptions if anything is
        missing.
        """
        if isinstance(input_dict, dict):
            if not input_dict['reaction']:
                raise Exception("Empty Reaction")
            elif not isinstance(input_dict['reaction'], str):
                raise Exception("Not a String!")
        else:
            raise Exception('input must be a json object.')
        self.reaction = input_dict['reaction']


def apply(input):
    """
    Predict Products of the Reaction (SMILES Notation)
    Returns the product as the output.
    """

    input = InputObject(input)

    num_layers = 4
    d_model = 256
    num_heads = 8
    dff = 1024
    vocab_size = 193
    pe_input = 130
    pe_target = 80
    rate = 0.1

    # Creating Weights Object
    client = Algorithmia.client('simtds2YG9Ed/wd5xucmvHy+U8G1')
    weights = client.file('data://.my/TFModel/tf_MIT400K_mixed.h5')
    # Model Loading
    model = Transformer(num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target, rate)
    model.load_weights(weights.path)
    tokenizer = re.compile(
        r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|se?|p|\(|\)|\.|=|#|'
        r'-|\+|\\\\|\/|:|~|@|\?|>|>>|\*|\$|\%[0-9]{2}|[0-9])'
    )
    pred = Prediction(model, tokenizer, max_reg=pe_input, max_prod=pe_target, beam_size=1, reduce=False)
    answers = pred.prediction(input.reaction)
    output = {'product': answers}
    return output

