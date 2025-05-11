import os
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
import time
from tfmapp.images_federation.utils.task_images import (
    get_model,
    get_weights,
    set_weights,
    load_image_data,
    train,
    test,
)

class FlowerClient(NumPyClient):
    """Federated Image Classification Client using ResNet18."""

    def __init__(self, model, trainloader, valloader, testloader, local_epochs, lr, batch_size, device, wandb_enabled, target_delta, noise_multiplier, max_grad_norm, use_dp, use_sa, is_demo, timeout):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.wandb_enabled = wandb_enabled

        # Differential Privacy parameters 
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.use_dp = use_dp

        # SA parameters
        self.use_sa = use_sa
        self.is_demo = is_demo
        self.timeout = timeout

    def get_parameters(self, config):
        """Get local model parameters."""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """Local training and sending of the updated model."""
        print("[CLIENT] Starting local training...")

        model = self.model
        set_weights(model, parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        privacy_engine = PrivacyEngine(secure_mode=False)   
        if self.use_dp:
            (
                model,
                optimizer, # Here I convert the optimizer into DP-SGD (Differentially Private Stochastic Gradient Descent)
                self.trainloader,
            ) = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        
        if not self.is_demo:
            history = train(
                self.model,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.device,
                optimizer,
            )

            last_epoch = -1
            train_loss = history["train_loss"][last_epoch]
            val_loss = history["val_loss"][last_epoch]
            train_acc = history["train_accuracy"][last_epoch]
            val_acc = history["val_accuracy"][last_epoch]
            train_recall = history["train_recall"][last_epoch]
            val_recall = history["val_recall"][last_epoch]
            train_f1 = history["train_f1"][last_epoch]
            val_f1 = history["val_f1"][last_epoch]
            train_prec = history["train_precision"][last_epoch]
            val_prec = history["val_precision"][last_epoch]

            print(f"📊 [CLIENT] Final Training Metrics:")
            print(f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
            print(f"Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} Prec: {val_prec} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
            print("Confusion Matrix (Train):")
            print(history["train_conf_matrix"])
            print("Confusion Matrix (Val):")
            print(history["val_conf_matrix"])
            print("-" * 60)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_recall": train_recall,
                "val_recall": val_recall,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_precision": train_prec,
                "val_precision": val_prec,
            }
        
         # Force a significant delay for testing purposes
        if self.is_demo:
            if config.get("drop", False):
                print(f"Client dropped for testing purposes.")
                time.sleep(self.timeout)
            else:
                print(f"Client uploading parameters: {get_weights(self.mlp)[0].flatten()[:3]}...")

        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print("🧪 [CLIENT] Evaluating global model on local test data...")
        set_weights(self.model, parameters)
        loss, accuracy, recall, f1, precision, conf_matrix = 0.0, 0.0, 0.0, 0.0, 0.0, None
        if not self.is_demo:
            loss, accuracy, recall, f1, precision, conf_matrix = test(
                self.model, self.testloader, self.device
            )

            print(f"Test -> Loss: {loss:.4f} | Acc: {accuracy:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Precision: {precision:.4f}")
            print("Confusion Matrix (Test):")
            print(conf_matrix)
            print("-" * 60)

            metrics = {
                "test_accuracy": accuracy,
                "test_recall": recall,
                "test_f1": f1,
                "test_loss": loss,
                "test_precision": precision,
            }

        return loss, len(self.testloader.dataset), metrics


# Required for Flower runtime
def client_fn(context: Context):
    dataset_path = context.node_config["dataset-path"]
    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    wandb = context.run_config["use-wandb"]
    use_dp = context.run_config["use-dp"]
    use_sa = context.run_config["use-sa"]
    is_demo = context.run_config["is-demo"]
    timeout = context.run_config["timeout"]
    target_delta = context.run_config["target-delta"]
    noise_multiplier = context.run_config["noise-multiplier"]
    max_grad_norm = context.run_config["max-grad-norm"]
    device = context.run_config["device"]
    print(f"EL CLIENTE ESTA USANDO {device}")

    trainloader, valloader, testloader, num_classes, _ = load_image_data(dataset_path, batch_size)
    model = get_model(num_classes)

    return FlowerClient(
        model,
        trainloader,
        valloader,
        testloader,
        local_epochs,
        lr,
        batch_size,
        device,
        wandb_enabled=wandb,
        target_delta=target_delta,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        use_dp=use_dp,
        use_sa=use_sa,
        is_demo=is_demo,
        timeout=timeout,
    ).to_client()


# Register Flower ClientApp
app = ClientApp(client_fn)

# app = ClientApp(
#     client_fn=client_fn,
#     mods=[
#         secaggplus_mod,
#     ],
# )
