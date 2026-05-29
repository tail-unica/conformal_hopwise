# @Time   : 2026
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

"""hopwise.model.conformal_calibration
#############################
Model calibration using conformal prediction (Conformal Risk Control)

source: https://arxiv.org/abs/2208.02814, https://github.com/aangelopoulos/conformal-risk

"""

import numpy as np
import torch
from tqdm import tqdm

from hopwise.utils import set_color


class AbstractCalibration:
    """
    Abstract class for calibration. It contains the common code for the calibration process, such as loading the data and the trainer, and defining the parameters for the conformal risk control. The actual calibration process is implemented in the Calibration class, which inherits from this abstract class.
    """  # noqa: E501

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        self.config = config
        self.logger = logger
        self.metric = metric
        self.crc_config = config["calibration"]
        self.topk = max(config["topk"])


        # data
        self.train_data = calibration_data["train"]
        self.calib_data = calibration_data["calibration"]
        self.test_data = calibration_data["test"]
        self.full_test_data = calibration_data["full_test"]

        # crc parameters
        self.alpha = self.crc_config["alpha"]
        # self.delta = self.crc_config['delta']
        self.n_lambdas = self.crc_config["n_lambdas"]
        self.maximum_metric_score = self.crc_config["maximum_metric_score"]

        # calibration parameters for expectation check
        self.n_trials = self.crc_config["n_trials"]
        self.n_calibration_users = int(
            (self.full_test_data.dataset.user_num - 1) * self.crc_config["n_calibration_users"]
        )

        # trainer and evaluation config
        self.trainer = trainer
        self.evaluation_config = eval_config

    def forward(self, data, **kwargs):
        raise NotImplementedError

    def calibrate(self):
        raise NotImplementedError


class ConformalRiskControl(AbstractCalibration):
    """Calibration is used to calibrate the model using conformal risk control."""

    def __init__(self, config, metric, trainer, logger, eval_config, calibration_data):
        super().__init__(config, metric, trainer, logger, eval_config, calibration_data)

    def get_lambdas(self, conf_scores):
        """
        Return the thresholds to use to test the risk constraint. Instead of using all the possible predicted score by the full_sort_prediction function, we use a grid of lambda values choose randomly in a range between the minimum and the maximum values in conf_scores. # noqa: E501

        Args:
            conf_scores (n_users x topk size)): full_sort_prediction confidence scores at cut off k

        Returns:
            lambdas: possible thresholds to use to test the risk constraint, chosen randomly in a range between the minimum and the maximum values in conf_scores.
        """  # noqa: E501
        conf_scores = conf_scores.reshape(-1)
        conf_scores = conf_scores[torch.isfinite(conf_scores)]
        lambdas = torch.linspace(
            conf_scores.min(),
            conf_scores.max(),
            self.n_lambdas,
            device=conf_scores.device,
            dtype=conf_scores.dtype,
        )
        return lambdas

    def get_bound(self, calib_loss_table):
        """
        Given the loss table for the calibration set, we compute the risk estimate for each lambda threshold

        Args:
            calib_loss_table (users x n_lambdas): The loss table for the calibration set, where each row is a user and each column is a lambda threshold.

        Returns:
            bound: The risk estimate for each lambda threshold.
        """  # noqa: E501
        n = calib_loss_table.shape[0]
        rhat = calib_loss_table.mean(dim=0)
        bound = (n / (n + 1)) * rhat + (self.maximum_metric_score / (n + 1))
        return bound

    def select_lhat(self, bound, lambdas, verbose=True):
        feasible_mask = bound <= self.alpha
        feasible_idxs = torch.where(feasible_mask)[0]

        if feasible_idxs.numel() == 0:
            # Nothing satisfies risk: use smallest lambda (largest set) as safest fallback.
            lhat_idx = 0

            if verbose:
                self.logger.warning(
                    set_color(
                        f"No lambda satisfies the risk constraint; using smallest lambda "
                        f"{lambdas[lhat_idx]:.4f} with bound {bound[lhat_idx]:.4f}",
                        "red",
                    )
                )
            return lambdas[lhat_idx], lhat_idx

        # Monotone-style candidate: rightmost feasible index.
        rightmost_feasible_idx = feasible_idxs.max()

        # Robust candidate for non-monotone bounds: highest feasible bound value.
        feasible_bounds = torch.where(feasible_mask, bound, torch.full_like(bound, -torch.inf))
        best_feasible_idx = torch.argmax(feasible_bounds)

        if rightmost_feasible_idx != best_feasible_idx:
            if verbose:
                self.logger.warning(
                    set_color(
                        "Non-monotone feasible bounds detected: selecting argmax(bound | bound<=alpha) "
                        f"at idx {best_feasible_idx.item()} (bound={bound[best_feasible_idx]:.4f}) "
                        f"instead of rightmost feasible idx {rightmost_feasible_idx.item()} "
                        f"(bound={bound[rightmost_feasible_idx]:.4f}).",
                        "yellow",
                    )
                )
            lhat_idx = best_feasible_idx
        else:
            lhat_idx = rightmost_feasible_idx

        return lambdas[lhat_idx], lhat_idx

    def forward(self, data, **kwargs):
        """
            Make a forward pass on the data using the trainer, and return the results, the losses and the confidence scores. If a threshold is provided, apply it to the confidence scores to get the calibrated results.

        Args:
            data: dataloader to evaluate on
            threshold (_type_, optional): Threshold to apply to the confidence scores to get the calibrated results. If None, return the uncalibrated results. Defaults to None.

        Returns:
            result: The metric results of the evaluation.
            losses: The losses for each user and each cut off k.
            confidence_scores: The confidence scores for each user and each cut off k.
        """  # noqa: E501
        # update self.evaluation config with kwargs
        for key, value in kwargs.items():
            self.evaluation_config[key] = value

        return self.trainer.evaluate(data, **self.evaluation_config)

    def get_loss_and_size_tables(self, conf_scores, lambdas):
        """
            Given the confidence scores, the losses and the lambda thresholds, compute the loss and size tables for each lambda threshold. The loss table contains the loss for each example and each lambda greater than the threshold, while the size table contains the size of the predicted set for each example and each lambda threshold.

            in case the higher the confidence score, the better, then you should have k = np.sum(conf_scores >= lmbda, axis=1). Otherwise, in case the higher the confidence score, the worse, then you should have k = np.sum(conf_scores >= 1 - lmbda, axis=1)

        Args:
            conf_scores (n_users x topk size)): it contains the full_sort_prediction confidence scores at cut off k
            losses (n_users x topk size)): it contains the metric score for each user and each cut off k
            lambdas : possible thresholds

        Returns:
            loss_table: (n_users x n_lambdas) metric value greater than the threshold for each user and each lambda
            size_table: (n_users x n_lambdas) size of the predicted set for each user and each lambda
        """  # noqa: E501
        
        loss_table = torch.zeros((len(conf_scores), len(lambdas)), device=conf_scores.device, dtype=conf_scores.dtype)
        size_table = torch.zeros((len(conf_scores), len(lambdas)), device=conf_scores.device, dtype=conf_scores.dtype)
        for i, lmbda in enumerate(tqdm(lambdas, desc="Computing loss and size tables for each lambda threshold")):
            # Lambda defines the set: keep only scores above threshold.
            mask = conf_scores >= lmbda
            conf_scores_masked = torch.where(mask, conf_scores, -torch.inf)

            # topk indices and values (descending)
            topk = torch.topk(conf_scores_masked, self.topk, dim=-1)

            self.metric.set_metric_data(conf_scores_masked, topk.indices)

            # Per-user metric curve over prefix sizes (k = 1..topk).
            results = self.metric.calculate_metric()[:, -1]
            valid_scores = torch.isfinite(topk.values)
            set_sizes = valid_scores.sum(dim=1)
            size_table[:, i] = set_sizes.to(size_table.dtype)

            loss_table[:, i] = results.to(loss_table.dtype)

        return loss_table, size_table

    def calibrate(self):
        """
        Given the losses and confidence scores for the calibration set, compute the lambda threshold lhat that satisfies the risk constraint, and return it along with the estimated risk and average size of the predicted sets at that threshold.

        Args:
            calib_losses (users x topk size): Metric scores at cut-off k
            confidence_scores (users x topk): full_sort_predict scores

        Returns:
            lhat: The lambda threshold that satisfies the risk constraint.
            risk: The estimated risk (average of the metric scores) at the chosen lambda threshold.
            avg_size: The average size of the topk after applying the lambda threshold lhat (i.e., the average number of items in the predicted set for each user after applying the threshold).
        """  # noqa: E501
        self.logger.info(set_color(f"Running evaluation on calibration set", "green"))
        results, confidence_scores, positive_u_list, positive_i_list, struct = self.forward(self.calib_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list, struct)
        confidence_scores = self.metric.process_scores(confidence_scores)

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        # calculate CRC upper bound
        bound = self.get_bound(loss_table)

        # select lambda threshold lhat that satisfies the risk constraint at the alpha level
        lhat, lhat_idx = self.select_lhat(bound, lambdas, verbose=True)

        # return logs
        self._logging(lhat, lhat_idx, bound, loss_table, size_table, lambdas)
        
        self.logger.info(set_color(f"Calibrating test results", "green"))
        calibrated_results, _, _, _, _ = self.forward(self.test_data, threshold=lhat)

        return results, calibrated_results

    def _logging(self, lhat, lhat_idx, bound, loss_table, size_table, lambdas):
        # get the risk and size at the chosen lambda threshold lhat
        risk = loss_table[:, lhat_idx].mean()
        avg_size = size_table[:, lhat_idx].mean()

        if lhat_idx == 0:
            self.logger.info(
                set_color(
                    f"No lambda satisfies the risk constraint; using smallest lambda "
                    f"{lambdas[lhat_idx]:.4f} with bound {bound[lhat_idx]:.4f}",
                    "red",
                )
            )
            self.logger.info(
                set_color(
                    f"[CALIBRATION STEP] Risk at min lambda (default without crc) {lambdas[0]:.4f} is {loss_table[:, 0].mean():.4f} with bound {bound[0]:.4f} and average topk size {size_table[:, 0].mean():.2f}\n",  # noqa: E501
                    "red",
                )
            )
        else:
            self.logger.info(
                set_color(
                    f"[CALIBRATION STEP] Calibrating results with conformal risk control (threshold={lhat:.4f}), risk {risk:.4f} with min bound {bound.min():.4f} and max bound {bound.max():.4f}, average topk size (inefficiency) {avg_size:.2f}",  # noqa: E501
                    "green",
                )
            )


    def prove_expectation(self):
        """
        Prove that the risk is controlled in expectation by running multiple trials of the calibration process. In each trial, we randomly split the full test set into a calibration user set and a test user set, compute the lambda threshold lhat on the calibration set, and then compute the risk and average size on the test set using that lhat. Finally, we average the risk and size across trials to get an estimate of the expected risk and size.
        """  # noqa: E501

        self.logger.info(set_color(f"Running evaluation for expectation check", "green"))
        _, confidence_scores, positive_u_list, positive_i_list, struct = self.forward(self.full_test_data)
        self.metric.set_positive_data(positive_u_list, positive_i_list, struct)
        confidence_scores = self.metric.process_scores(confidence_scores)

        # get the lambda thresholds to test for the risk constraint
        lambdas = self.get_lambdas(confidence_scores)

        loss_table, size_table = self.get_loss_and_size_tables(confidence_scores, lambdas)

        risks, lhats, sizes = list(), list(), list()
        for _ in tqdm(
            range(self.n_trials),
            desc=f"Computing conformal risk control in expectation using {self.crc_config['n_trials']} trials",
        ):
            lhat, risk, size = self.compute_trial(loss_table, size_table, lambdas)
            risks.append(risk.cpu().item())
            lhats.append(lhat.cpu().item())
            sizes.append(size.cpu().item())

        self.logger.info(set_color(f"Average risk across trials = {np.mean(risks):.3f}", "red"))
        self.logger.info(set_color(f"Average threshold across trials = {np.mean(lhats):.3f}", "red"))
        self.logger.info(set_color(f"Average size across trials = {np.mean(sizes):.3f}", "red"))

    def compute_trial(self, loss_table, size_table, lambdas):
        """Compute risk and sizes for a trial.

        Args:
            loss_table: [num_examples, num_lambdas] losses by lambda from small to large.
            size_table: [num_examples, num_lambdas] sizes by lambda from small to large.
            lambdas: [num_lambdas] lambda values from small to large.

        Returns:
            lhat: Confidence score threshold.
            avg_loss: Average set loss.
            avg_size: Average set size.
        """
        # Split to calibration and test.
        perm = torch.randperm(loss_table.shape[0], device=loss_table.device)
        loss_table = loss_table.index_select(0, perm)
        size_table = size_table.index_select(0, perm)
        calib_loss_table = loss_table[: self.n_calibration_users]
        
        valid_loss_table = loss_table[self.n_calibration_users :]
        valid_size_table = size_table[self.n_calibration_users :]

        # Compute threshold
        bound = self.get_bound(calib_loss_table)
        lhat, lhat_idx = self.select_lhat(bound, lambdas, verbose=False)

        # Compute losses and size.
        avg_loss = valid_loss_table[:, lhat_idx].mean()
        avg_size = valid_size_table[:, lhat_idx].mean()

        return lhat, avg_loss, avg_size
