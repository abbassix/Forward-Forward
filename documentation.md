## ff_model
`Class FF_model`: initializes the `FF_model` by setting up the forward-forward model layers, peer normalization mechanism, downstream linear classifier using the outputs of all hidden layers, and the corresponding loss functions for forward-forward and classification tasks, followed by weight initialization.

1. `__init__`: initializes the models:
    - `opt`: configuration options passed during initialization.
    - `num_channels`: a list representing the number of hidden units in each layer.
    - `act_fn`: activation function for the model.
    - `model` has `1` input layer and `num_layers - 1` hidden layers:
        - input layer: `Linear(in_features=784, out_features=hidden_dim, bias=True)` for MNIST input (784 pixels).
        - hidden layers: `Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)` each.
    - `ff_loss`: forward-forward loss function (binary cross-entropy with logits).
    - `running_means`: list of tensors for peer normalization, initialized to `0.5` for each hidden layer.
    - `linear_classifier` is the downstream classification model. It has `2000` inputs since the model uses output of all hidden layers as the input for classification.
        - `Linear(in_features=2000, out_features=10, bias=False)`
    - `running_means`: initializes a list of running mean tensors, each filled with `0.5`, with sizes corresponding to the number of channels per layer and placed on the specified device.
    - `classification_loss`: loss function for the downstream classification model (cross-entropy).
2. `_init_weights`: initializes the weights:
    - `model` weights are drawn from a normal distribution with mean zero and standard deviation proportional to the inverse square root of the input dimensions.
    - `linear_classifier` weights are initialized to zeros.
3. `_layer_norm`: normalizes the input `z` by dividing it by the square root of the mean of its squared values along the last dimension, with a small epsilon added for numerical stability.
4. `_calc_peer_normalization_loss`: calculates the peer normalization loss by updating the `running_means` of the activations for positive samples, and computes the loss as the squared difference between the average of the running means and the running means themselves.
5. `_calc_ff_loss`: computes the forward-forward loss by calculating the sum of squares for the activations `z` and comparing it against a threshold using binary cross-entropy. Also returns the accuracy of the FF predictions.
    - `z`: This is the output from one of the model's hidden layers, which has undergone some transformation and activation.
    - `labels`: This contains the labels corresponding to the inputs passed through the network. Typically, the labels are binary (1 for positive samples and 0 for negative samples).

## `FF_model` class:


6. `forward`: defines the forward pass through the model.
    - Concatenates positive and negative samples.
    - Applies layer normalization and feeds the input through each hidden layer.
    - Updates the peer normalization loss and forward-forward loss at each layer.
    - Detaches the output at each layer before normalization.
    - Computes and aggregates all loss components.

7. `forward_downstream_classification_model`: passes the neutral samples through the hidden layers, gathers the outputs from the hidden layers, and feeds them into the downstream classification model.
    - Computes classification loss and accuracy.
    - Aggregates loss into the total scalar outputs.## `FF_model` class:
The model trained with Forward-Forward (FF).

1. `__init__`: initializes the model.
    - `opt`: configuration options passed during initialization.
    - `num_channels`: a list representing the number of hidden units in each layer.
    - `act_fn`: activation function for the model.
    - `model`: 
        - input layer: `Linear(in_features=784, out_features=hidden_dim, bias=True)` for MNIST input (784 pixels).
        - hidden layers: `Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)` each.
    - `ff_loss`: forward-forward loss function (binary cross-entropy with logits).
    - `running_means`: list of tensors for peer normalization, initialized to `0.5` for each hidden layer.
    - `linear_classifier`: downstream classification model using the outputs of all hidden layers, with input size `sum(hidden_dim)`.
        - `Linear(in_features=total_hidden_dim, out_features=10, bias=False)`.
    - `classification_loss`: loss function for the downstream classification model (cross-entropy).
    - `_init_weights`: initializes the weights using normal distribution for `model` and zeros for the classifier.

2. `_init_weights`: initializes the weights:
    - `model` weights are drawn from a normal distribution with mean zero and standard deviation proportional to the inverse square root of the input dimensions.
    - Classifier weights are initialized to zeros.

3. `_layer_norm`: normalizes the input `z` by dividing it by the square root of the mean of its squared values along the last dimension, with a small epsilon added for numerical stability.

4. `_calc_peer_normalization_loss`: calculates the peer normalization loss by updating the running mean of the activations for positive samples and computing the squared difference between the average of the running means and the running means themselves.

5. `_calc_ff_loss`: computes the forward-forward loss by calculating the sum of squares for the activations `z` and comparing it against a threshold using binary cross-entropy. Also returns the accuracy of the FF predictions.

6. `forward`: defines the forward pass through the model.
    - Concatenates positive and negative samples.
    - Applies layer normalization and feeds the input through each hidden layer.
    - Updates the peer normalization loss and forward-forward loss at each layer.
    - Detaches the output at each layer before normalization.
    - Computes and aggregates all loss components.

7. `forward_downstream_classification_model`: passes the neutral samples through the hidden layers, gathers the outputs from the hidden layers, and feeds them into the downstream classification model.
    - Computes classification loss and accuracy.
    - Aggregates loss into the total scalar outputs.