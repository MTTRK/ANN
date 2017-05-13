from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense
import numpy as np
from itertools import chain
import morph_io as mio
from option import MainOption
from context import MainContext
from context import Context
from context import Prediction
from context import EvaluationContext
import evaluate as eval


def extend_word(word: str, ctx: Context):
    """
    Extends the word with symbols and returns them as a list
    :param word: to be extended 'abounded'
    :param ctx: Context holding the parameters
    :return: ['<s>','a','b','o','u','n','d','e','d','</s>']
    """
    symbols = [s for s in word]
    symbols.insert(0, ctx.START)
    symbols.append(ctx.STOP)
    return symbols


def predict_words(model, ctx: MainContext):
    """
    :param model: NeuralNetwork
    :param ctx: MainContext containing runtime parameters
    :return: [Prediction("ablatives", ['B', ...]), Prediction(...), ...]
    """
    predictions = []
    for word in ctx.test:
        _word = word.strip(' \n')
        current = predict_word(model, _word, ctx)
        predictions.append(Prediction(_word, current))
    return predictions


def predict_word(model, word: str, ctx: MainContext):
    """
    :param model: NeuralNetwork (Keras) model
    :param word: str (ex.: 'abounded')
    :param ctx: MainContext containing runtime parameters
    :return ['B', 'E', 'B', ...]
    """
    # PREP: abounded --> [<s>,a,b,o,u,n,d,e,d,</s>]
    symbol_list = extend_word(word, ctx)

    # WINDOW: [<s>,a,b,o,u,n,d,e,d,</s>] --> one-hot matrix of
    # 3 symbol windows generated from [<s>,<s>,a,b,o,u,n,d,e,d,</s>,</s>]
    symbol_matrix = mio.transform_input(symbol_list, ctx)
    prediction_matrix = model.predict(symbol_matrix)

    return mio.predictions_to_symbols(prediction_matrix, ctx)


def create_network(seed: int, layers: list):
    """
    :param seed: int
    :param layers: [Dense(...), ...]
    """
    np.random.seed(seed)
    network = Sequential()
    for layer in layers:
        network.add(layer)
    return network


def create_layers(indim: int, outdim: int, init: str, activation: str, hidden: int):
    """
    :param indim: number of input nodes
    :param outdim: number of output nodes
    :param init: name of initialization function (ex.: uniform)
    :param activation: name of activation function (ex.: relu)
    :param hidden: number of hidden layers in the network
    :return: [Layer, Layer, ...] list of Layer implementations
    """
    layers = [Dense(input_dim=indim, units=indim, kernel_initializer=init, activation=activation)]
    layers.extend([Dense(units=indim, kernel_initializer=init, activation=activation) for i in range(0, hidden)])
    layers.append(Dense(units=outdim, kernel_initializer=init, activation=activation))
    return layers


def build_train_model(ctx: MainContext, verbose: int = 1):
    """
    :param ctx: MainContext containing runtime parameters (hyperparameters)
    :param verbose: verbosity of the training process
    :return NeuralNetwork model
    """
    # [ab,cd] --> ['<s>','a','b','</s>','<s>','c','d','</s>']
    input_symbol_list = list(chain.from_iterable([extend_word(seg.word, ctx) for seg in ctx.training]))
    input_matrix = mio.transform_input(input_symbol_list, ctx)

    # [[B,M],[B,M]] --> [B,M,B,M]
    output_symbol_list = list(chain.from_iterable([seg.seg_mapping for seg in ctx.training]))
    output_matrix = mio.transform_output(output_symbol_list, ctx)

    # create and train NN
    indim = len(input_matrix[0])
    outdim = len(output_matrix[0])
    batches = len(input_matrix)
    callbacks = [EarlyStopping(monitor='val_loss', patience=ctx.earlystop)]

    model = create_network(
        seed=561,
        layers=create_layers(indim, outdim, ctx.init, ctx.activate, ctx.hiddenlayer))

    model.compile(loss=ctx.loss, optimizer=ctx.optimize, metrics=['accuracy'])

    model.fit(input_matrix, output_matrix,
              epochs=ctx.epochs, batch_size=batches, callbacks=callbacks,
              validation_split=0.1, verbose=verbose)

    return model


def do_train_and_predict(ctx: MainContext):
    """
    Runs normal train & predict session
    :param ctx: contains the hyper-parameters
    """
    ctx.training = mio.read_goldstd(ctx.training, ctx)
    ctx.test = mio.read_file(ctx.test)

    model = build_train_model(ctx, 0)
    predictions = predict_words(model, ctx)
    mio.output_predictions(predictions, ctx)


def do_benchmark(ctx: MainContext):
    """
    Does benchmarking for finding the best set of parameters
    :param ctx: contains the hyper-parameters
    """
    ctx.training = mio.read_goldstd(ctx.training, ctx)
    ctx.devel = [segment.seg_mapping
                 for segment in mio.read_goldstd(ctx.devel, ctx)]
    ctx.test = list(mio.read_file(ctx.test))

    print('=== Benchmark ===\n')
    for window_size in [2, 3, 4]:
        for window_type in [0, 1, 2]:
            for hidden_layer in [1, 2, 3]:
                for epochs in [200, 300, 400]:
                    for activation in ['sigmoid', 'relu', 'tanh', 'softmax']:
                        for optimizer in ['adam', 'rmsprop']:
                            for loss in ['mean_squared_error', 'binary_crossentropy']:
                                ctx.windowsize = window_size
                                ctx.windowtype = window_type
                                ctx.hiddenlayer = hidden_layer
                                ctx.epochs = epochs
                                ctx.activate = activation
                                ctx.optimize = optimizer
                                ctx.loss = loss

                                model = build_train_model(ctx, verbose=0)
                                predictions = predict_words(model, ctx)

                                eval_context = EvaluationContext()
                                eval_context.actual = [pred.prediction for pred in predictions]
                                eval_context.expected = ctx.devel

                                aggr_metric = eval.get_aggregated_metric(
                                                eval.generate_metrics(eval_context))

                                print(str(ctx) +
                                      '--> F-Score=' + str(aggr_metric.get_fscore()) +
                                      ' Precision=' + str(aggr_metric.get_precision()) +
                                      ' Recall=' + str(aggr_metric.get_recall()),
                                      flush=True)


def main():
    option = MainOption()
    context = MainContext(
        _activate=option.activate,
        _earlystop=option.earlystop,
        _epochs=option.epochs,
        _hiddenlayer=option.hiddenlayer,
        _init=option.init,
        _loss=option.loss,
        _windowsize=option.windowsize,
        _windowtype=option.windowtype,
        _optimize=option.optimize,
        _devel=option.devel,
        _training=option.training,
        _test=option.words
    )
    if option.bmes:
        context.set_bmes_context()

    if option.benchmark:
        do_benchmark(context)
    else:
        do_train_and_predict(context)


if __name__ == "__main__":
    main()
