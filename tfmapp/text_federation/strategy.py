import json
import time
from logging import INFO
from pathlib import Path
import csv
import torch
import wandb
from flwr.common import logger, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.common.typing import UserConfig

PROJECT_NAME = "TFM-Text-Federated-Classification"

class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality that I have added"""

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_config = run_config
        self.start_time = time.time()
        self.use_wandb = use_wandb
        self.save_path, self.run_dir = self._create_run_dir(run_config)

        if self.use_wandb:
            self._init_wandb()

        # self.best_acc_so_far = 0.0
        self.results = {}

    def _init_wandb(self):
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _create_run_dir(self, config: UserConfig):
        current_time = time.strftime("%Y-%m-%d/%H-%M-%S")
        run_dir = current_time
        save_path = Path.cwd() / "tfmapp/images_federation/outputs" / run_dir
        save_path.mkdir(parents=True, exist_ok=False)

        with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
            json.dump(config, fp)

        return save_path, run_dir

    def _store_results(self, tag: str, results_dict):
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _export_fit_metrics_csv(self, server_round: int, client_metrics: dict):
        """Save client metrics to CSV for each round."""
        csv_file = Path(self.save_path) / "fit_client_metrics.csv"
        file_exists = csv_file.exists()

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "round", "client_id",
                    "train_loss", "train_accuracy", "train_recall", "train_f1", "train_precision",
                    "val_loss", "val_accuracy", "val_recall", "val_f1", "val_precision"
                ])

            for client_id, metrics in client_metrics.items():
                writer.writerow([
                    server_round, client_id,
                    metrics.get("train_loss"), metrics.get("train_accuracy"), metrics.get("train_recall"),
                    metrics.get("train_f1"), metrics.get("train_precision"),
                    metrics.get("val_loss"), metrics.get("val_accuracy"), metrics.get("val_recall"),
                    metrics.get("val_f1"), metrics.get("val_precision")
                ])

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})
        if self.use_wandb:
            wandb_log_dict = {
                f"Federated Evaluation/{k.replace('federated_evaluate_', '')}": v
                for k, v in results_dict.items()
                if k != "round"
            }
            wandb.log(wandb_log_dict, step=server_round)


    def aggregate_fit(self, server_round, results, failures):
        parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        self.parameters = parameters

        client_metrics_dict = {}
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics or {}
            client_metrics_dict[client_id] = metrics

            client_display_name = f"Client {client_id}"

            # ✅ W&B Logging por cliente
            if self.use_wandb:
                wandb.log({f"{client_display_name}/train_loss": float(metrics.get("train_loss", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/val_loss": float(metrics.get("val_loss", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/train_acc": float(metrics.get("train_accuracy", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/val_acc": float(metrics.get("val_accuracy", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/train_recall": float(metrics.get("train_recall", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/val_recall": float(metrics.get("val_recall", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/train_f1": float(metrics.get("train_f1", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/val_f1": float(metrics.get("val_f1", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/train_prec": float(metrics.get("train_precision", 0.0))}, step=server_round)
                wandb.log({f"{client_display_name}/val_prec": float(metrics.get("val_precision", 0.0))}, step=server_round)
        
        self._export_fit_metrics_csv(server_round, client_metrics_dict)
        return parameters, aggregated_metrics



    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Save and log results (all metrics including acc, prec, recall, f1)
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )

        # Report total training time after last round
        if server_round == self.run_config["num-server-rounds"]:
            elapsed = time.time() - self.start_time
            logger.log(INFO, f"Total training time: {elapsed:.2f} seconds")

        return loss, metrics
