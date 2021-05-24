import Algorithmia
import tensorflow as tf
from .ScaledTransformer import Transformer
from .BeamPredictor import Prediction
from .ModelSpace import SmilesModel, CgrModel

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


# Configure Tensorflow to only use up to 50% of the GPU.
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
            elif not isinstance(input_dict['beamWidth'], int):
                raise Exception("Not an Number!")
        else:
            raise Exception('input must be a json object.')
        self.reaction = input_dict['reaction']
        self.beam_width = input_dict['beamWidth']
        self.model = input_dict['model']


def apply(input):
    """
    Predict Products of the Reaction (SMILES Notation)
    Returns the product as the output.
    """

    input = InputObject(input)

    # Model Selecting
    models = {"smiles": SmilesModel, "cgr": CgrModel}
    params = models[input.model]

    # Initializing client for pretrained weights
    client = Algorithmia.client()

    # Model Loading
    model = Transformer(params.num_layers, params.d_model, params.num_heads, params.dff,
                        params.vocab_size, params.pe_input, params.pe_target, params.rate)
    model.load_weights(client.file(params.weights_uri).getFile().name)

    pred = Prediction(model, params.vocab, params.tokenizer, max_reg=params.pe_input, max_prod=params.pe_target,
                      beam_size=input.beam_width, reduce=params.reduce_smiles)
    answers = pred.prediction(input.reaction)
    output = {'product': answers}
    return output
