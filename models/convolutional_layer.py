class ConvolutionalLayer:
    def __int__(self, num_filters, filters_size, learning_rate,
                num_channels, activation_function) -> None:
        self.num_filters = num_filters
        self.filters_size = filters_size
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.activation_function = activation_function

    def forward_propagation(self, input_values):
