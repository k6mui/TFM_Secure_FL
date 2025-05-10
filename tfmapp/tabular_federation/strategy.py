import json
from logging import INFO
import torch
import wandb
from tfmapp.tabular_federation.utils.task_tabular import create_run_dir, set_weights
import time
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
import csv
from pathlib import Path

PROJECT_NAME = "TFM-jorgeMonserrat-tabularFederation"

class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality that I have added
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config
        self.start_time = time.time()

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb

        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # # Keep track of best acc
        # self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _export_fit_metrics_csv(self, server_round: int, client_metrics: dict):
        """Guarda métricas por cliente y ronda en un CSV (una fila por cliente y ronda)."""
        csv_file = Path(self.save_path) / "fit_client_metrics.csv"
        file_exists = csv_file.exists()

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = [
                    "round", "client_id",
                    "train_loss", "train_accuracy", "train_recall", "train_f1", "train_prec",
                    "val_loss", "val_accuracy", "val_recall", "val_f1", "val_prec"
                ]
                writer.writerow(headers)

            for client_id, metrics in client_metrics.items():
                row = [
                    server_round, client_id,
                    metrics.get("train_loss"), metrics.get("train_accuracy"),
                    metrics.get("train_recall"), metrics.get("train_f1"), metrics.get("train_prec"),
                    metrics.get("val_loss"), metrics.get("val_accuracy"),
                    metrics.get("val_recall"), metrics.get("val_f1"), metrics.get("val_prec"),
                ]
                writer.writerow(row)


    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """Store results and log them to W&B if enabled, formatting better grouping."""
        # Store results en JSON local
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if tag == "federated_evaluate":
            csv_file = Path(self.save_path) / "federated_eval_metrics.csv"
            file_exists = csv_file.exists()

            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    headers = ["round"] + list(results_dict.keys())
                    writer.writerow(headers)
                writer.writerow([server_round] + list(results_dict.values()))

        if self.use_wandb:
            wandb_log_dict = {
                f"Federated Evaluation/{k.replace('federated_evaluate_', '')}": v
                for k, v in results_dict.items()
                if k != "round"
            }
            wandb.log(wandb_log_dict, step=server_round)

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )

        # Si es la última ronda, medir tiempo total
        if server_round == self.run_config["num-server-rounds"]:
            end_time = time.time()
            total_time = end_time - self.start_time
            logger.log(INFO, f"Total training time: {total_time:.2f} seconds")


        return loss, metrics

    def aggregate_fit(self, server_round, results, failures):
        parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        client_metrics_dict = {}
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            client_metrics_dict[client_id] = metrics

            client_display_name = f"Client with id: {client_id}"

            # Logear a W&B individualmente por cliente
            if self.use_wandb:
                wandb.log({f"{client_display_name}/train_loss": metrics.get("train_loss", None)}, step=server_round)
                wandb.log({f"{client_display_name}/val_loss": metrics.get("val_loss", None)}, step=server_round)
                wandb.log({f"{client_display_name}/train_acc": metrics.get("train_accuracy", None)}, step=server_round)
                wandb.log({f"{client_display_name}/val_acc": metrics.get("val_accuracy", None)}, step=server_round)
                wandb.log({f"{client_display_name}/train_recall": metrics.get("train_recall", None)}, step=server_round)
                wandb.log({f"{client_display_name}/val_recall": metrics.get("val_recall", None)}, step=server_round)
                wandb.log({f"{client_display_name}/train_f1": metrics.get("train_f1", None)}, step=server_round)
                wandb.log({f"{client_display_name}/val_f1": metrics.get("val_f1", None)}, step=server_round)
                wandb.log({f"{client_display_name}/train_prec": metrics.get("train_prec", None)}, step=server_round)
                wandb.log({f"{client_display_name}/val_prec": metrics.get("val_prec", None)}, step=server_round)


        self._export_fit_metrics_csv(server_round, client_metrics_dict)

        return parameters, aggregated_metrics

