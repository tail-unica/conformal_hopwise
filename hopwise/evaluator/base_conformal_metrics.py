import torch

class AbstractConformalLoss:
    """:class:`AbstractConformalLoss` is the base object of all conformal losses. If you want to implement a conformal loss, you should inherit this class."""

    def __init__(self, config):
        self.topk = max(config["topk"])
        self.define_base_variables()

    def define_base_variables(self):
        """Define the base variables needed to calculate the metric during the calibration step."""
        self.positive_u = None
        self.positive_i = None
        self.struct = None
    
    def set_metric_data(self, scores, topk_indices):
        raise NotImplementedError("You should implement this method to set the data needed to calculate the metric during the calibration step. The data can be the scores, the topk indices, and the struct (the struct is a dictionary that contains all the data of the evaluation).")
    
    def set_positive_data(self, positive_u_list, positive_i_list, struct):
        """
        This method is used to set the positive data for the conformal loss. The positive data can be used to calculate the metric during the calibration step. The positive data is a list of user indices and a list of item indices that correspond to the positive interactions (true labels) in the calibration set. The method concatenates the lists if they are provided as lists or tuples, or directly assigns them if they are already tensors.
        """
        self.struct = struct

        if isinstance(positive_u_list, (list, tuple)):
            self.positive_u = torch.cat(positive_u_list, dim=0)
        else:
            self.positive_u = positive_u_list

        if isinstance(positive_i_list, (list, tuple)):
            self.positive_i = torch.cat(positive_i_list, dim=0)
        else:
            self.positive_i = positive_i_list
    
    def calculate_metric(self):
        raise NotImplementedError("You should implement this method to calculate the metric during the calibration step. The metric is calculated using the data set in the set_metric_data method and the positive data set in the set_positive_data method.")