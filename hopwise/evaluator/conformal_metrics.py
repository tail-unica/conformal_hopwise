# @Time   : 2026/03/01
# @Author : Alessandro Soccol
# @email  : alessandro.soccol@unica.it

"""hopwise.evaluator.conformal_metrics
#####################################
"""

import torch
from hopwise.evaluator.base_conformal_metrics import AbstractConformalLoss


class Miscoverage(AbstractConformalLoss):
    """
        Miscoverage is a calibration loss for conformal predictors. It is the share of
        instances where the true label falls outside the predicted set (false negatives).
        Calibration aims to control this rate at a target level (e.g., 0.20 for 80% recall).
        This illustrates how CRC reframes miscoverage in Split Conformal Prediction.
    """
    def __init__(self, config):
        super().__init__(config)

    def set_metric_data(self, scores, topk_indices):
        pos_matrix = torch.zeros_like(scores, dtype=torch.int)
        pos_matrix[self.positive_u, self.positive_i] = 1
        pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_indices)
        matrix = torch.cat((pos_idx, pos_len_list), dim=1)
        topk_idx, pos_len_list = torch.split(matrix, [self.topk, 1], dim=1)
        self._pos_index = topk_idx.to(torch.bool)
        self._pos_len = pos_len_list.squeeze(-1)
    
    def process_scores(self, scores):
        return scores[self.positive_u]
    
    def calculate_metric(self):
        return 1 - (torch.cumsum(self._pos_index, dim=1) / self._pos_len.reshape(-1, 1))


class Unwanted(AbstractConformalLoss):
    """
    Recall computed on user-specific unwanted items. It measures the share of
    unwanted items that appear in the recommendation list at the maximum $K$.
    """

    def __init__(self, config):
        super().__init__(config)

    def define_base_variables(self):
        super().define_base_variables()
        self.rec_items = None
        self._users_with_neg_feedback = None
        self._unwanted_items_full = None
        self._unwanted_items = None
        self._denom = None
        self._cached_device = None
    
    def set_positive_data(self, positive_u_list, positive_i_list, struct):
        super().set_positive_data(positive_u_list, positive_i_list, struct)
        # Drop padding user (index 0), keep only users with at least one unwanted item.
        unwanted_items = torch.as_tensor(struct.get("data.unwanted_items")[1:], dtype=torch.int)
        users_with_neg_feedback = unwanted_items.sum(dim=1) != 0

        self._users_with_neg_feedback = users_with_neg_feedback
        self._unwanted_items_full = unwanted_items
        self._unwanted_items = None
        self._denom = None
        self._cached_device = None
    
    def set_metric_data(self, scores, topk_indices):
        self.rec_items = topk_indices
        device = topk_indices.device

        if self._cached_device != device:
            users_mask = self._users_with_neg_feedback.to(device)
            self._unwanted_items = self._unwanted_items_full.to(device)[users_mask]
            self._denom = self._unwanted_items.sum(dim=1, keepdim=True)
            self._cached_device = device
    
    def process_scores(self, scores):
        return scores[self._users_with_neg_feedback.to(scores.device)]

    def calculate_metric(self):
        is_unwanted_rec = torch.gather(self._unwanted_items, dim=1, index=self.rec_items)
        count_unwanted = is_unwanted_rec.cumsum(dim=1)
        return count_unwanted / self._denom
