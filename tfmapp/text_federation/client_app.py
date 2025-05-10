import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine
import time
from flwr.client.mod import fixedclipping_mod, secaggplus_mod

from tfmapp.text_federation.utils.task_text import (
    get_model,
    get_weights,
    set_weights,
    load_data,
    train,
    test,
)

class FlowerTextClient(NumPyClient):
    """Federated NLP Client using nlpie/bio-tinybert for MedMCQA."""

    def __init__(self, model, trainloader, valloader, testloader, class_weights, local_epochs, device, wandb_enabled, target_delta, noise_multiplier, max_grad_norm, use_dp, use_sa, is_demo, timeout):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.class_weights = class_weights
        self.local_epochs = local_epochs
        self.device = device
        self.wandb_enabled = wandb_enabled

        # Differential Privacy parameters 
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # SA parameters
        self.use_sa = use_sa
        self.is_demo = is_demo
        self.timeout = timeout

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        print("[CLIENT] Starting local training...")

        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                self.class_weights,
            )

            last_epoch = -1
            metrics = {
                "train_loss": history["train_loss"][last_epoch],
                "val_loss": history["val_loss"][last_epoch],
                "train_accuracy": history["train_accuracy"][last_epoch],
                "val_accuracy": history["val_accuracy"][last_epoch],
                "train_precision": history["train_precision"][last_epoch],
                "val_precision": history["val_precision"][last_epoch],
                "train_recall": history["train_recall"][last_epoch],
                "val_recall": history["val_recall"][last_epoch],
                "train_f1": history["train_f1"][last_epoch],
                "val_f1": history["val_f1"][last_epoch],
            }

            print("\U0001F4CA [CLIENT] Final Training Metrics:")
            print(metrics)
            print("Confusion Matrix (Train):\n", history["train_conf_matrix"])
            print("Confusion Matrix (Val):\n", history["val_conf_matrix"])
            print("-" * 60)
        
        # Force a significant delay for testing purposes
        if self.is_demo:
            if config.get("drop", False):
                print(f"Client dropped for testing purposes.")
                time.sleep(self.timeout)
            else:
                print(f"Client uploading parameters: {get_weights(model)[0].flatten()[:3]}...")

        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        print("[CLIENT] Evaluating global model on local test data...")
        set_weights(self.model, parameters)
        loss, accuracy, recall, f1, precision, conf_matrix = 0.0, 0.0, 0.0, 0.0, 0.0, None
        if not self.is_demo:
            loss, acc, prec, rec, f1, conf_matrix = test(self.model, self.testloader, self.device)

            print(f"Test -> Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            print("Confusion Matrix (Test):\n", conf_matrix)
            print("-" * 60)

            metrics = {
                "test_loss": loss,
                "test_accuracy": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1": f1,
            }

        return loss, len(self.testloader.dataset), metrics


# Required for Flower runtime
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    wandb = context.run_config["use-wandb"]
    model_name = "nlpie/bio-tinybert"
    device = context.run_config["device"]
    use_dp = context.run_config["use-dp"]
    use_sa = context.run_config["use-sa"]
    is_demo = context.run_config["is-demo"]
    timeout = context.run_config["timeout"]
    target_delta = context.run_config["target-delta"]
    noise_multiplier = context.run_config["noise-multiplier"]
    max_grad_norm = context.run_config["max-grad-norm"]

    trainloader, valloader, num_labels, class_weights = load_data(partition_id, num_partitions, model_name)
    testloader = valloader  # For now use val set as test

    model = get_model(num_labels=num_labels, class_weights=class_weights)

    return FlowerTextClient(
        model,
        trainloader,
        valloader,
        testloader,
        class_weights,
        local_epochs,
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
