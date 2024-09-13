## ff_model
`FF_model` class:
- The model trained with Forward-Forward (FF).
1. `__init__`: initializes:
    - `num_channels` = [1000, 1000, 1000] where 1000 is the hidden layers' dimension and 3 is the number of hidden layers.
    - `model` has:
        - input layer: `Linear(in_features=784, out_features=1000, bias=True)`
        - hidden layers: `Linear(in_features=1000, out_features=1000, bias=True)` each (2, i.e., `num_layers - 1`)
        - output layer: `Linear(in_features=2000, out_features=10, bias=False)`; `in_features=2000` since the model uses output of all hidden layers as the input for classification.
