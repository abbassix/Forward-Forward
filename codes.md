## ff_model
`FF_model` class:
- The model trained with Forward-Forward (FF).
1. `__init__`: initializes:
    - `model` has:
        - input layer: `Linear(in_features=784, out_features=1000, bias=True)`
        - hidden layers: `Linear(in_features=1000, out_features=1000, bias=True)` each (2, i.e., `num_layers - 1`)
    - `linear_classifier` is the downstream classification model. It has `2000` inputs since the model uses output of all hidden layers as the input for classification.
        - `Linear(in_features=2000, out_features=10, bias=False)`
