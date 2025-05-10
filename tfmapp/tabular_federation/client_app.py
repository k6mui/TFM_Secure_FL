import torch
from tfmapp.tabular_federation.utils.task_tabular import MLP, get_weights, load_data, set_weights, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from opacus import PrivacyEngine
import time

class FlowerClient(NumPyClient):
    """
    Client class for Client set up in tabular data classification tasks.
    This class handles the local training and local evaluation of the model.
    """

    def __init__(self, mlp, trainloader, valloader, x_test, y_test, local_epochs, lr, batch_size, device, wandb, target_delta, noise_multiplier, max_grad_norm, use_dp, use_sa, is_demo, timeout):
        self.mlp= mlp
        self.trainloader = trainloader
        self.valloader = valloader
        self.x_test = x_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.wandb = wandb
        self.mlp.to(self.device)

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
        return get_weights(self.mlp)

    def fit(self, parameters, config):
        """Local training and sending of the updated model."""
        print("[CLIENT] Starting local training...")
        model = self.mlp
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
            # Ejecutar el entrenamiento local
            history, epsilon = train(
                model,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.batch_size,
                self.device,
                optimizer, 
                privacy_engine,
                self.target_delta,
                self.use_dp
            )

            if epsilon is not None:
                print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
            else:
                print("Epsilon value not available.")

            last_epoch = -1  # Último índice

            train_loss = history["train_loss"][last_epoch]
            val_loss = history["val_loss"][last_epoch]
            train_acc = history["train_accuracy"][last_epoch]
            val_acc = history["val_accuracy"][last_epoch]
            train_recall = history["train_recall"][last_epoch]
            val_recall = history["val_recall"][last_epoch]
            train_f1 = history["train_f1"][last_epoch]
            val_f1 = history["val_f1"][last_epoch]
            train_prec = history["train_prec"][last_epoch]
            val_prec = history["val_prec"][last_epoch]

            # Mostrar métricas por consola
            print(f"📊 [CLIENT] Métricas finales del entrenamiento local:")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f} | Prec: {train_prec}")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f} | Prec: {val_prec}")
            print("-" * 60)

            # También podrías imprimir la matriz de confusión si quieres
            print("Matriz de confusión (Train):")
            print(history["train_conf_matrix"])
            print("Matriz de confusión (Val):")
            print(history["val_conf_matrix"])

            # Empaquetar todas las métricas para que el servidor las reciba
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_recall": train_recall,
                "val_recall": val_recall,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_prec": train_prec,
                "val_prec": val_prec
            }

        # Force a significant delay for testing purposes
        if self.is_demo:
            if config.get("drop", False):
                print(f"Client dropped for testing purposes.")
                time.sleep(self.timeout)
            else:
                print(f"Client uploading parameters: {get_weights(model)[0].flatten()[:3]}...")

        return get_weights(model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the global model on local test data and send it back to the server for aggregated evaluation."""

        # Global weight assignment
        set_weights(self.mlp, parameters)
        
        # Get metrics
        print("[CLIENT] Evaluating global model on local test data...")
        loss, accuracy, recall, f1, precision, conf_matrix = 0.0, 0.0, 0.0, 0.0, 0.0, None
        if not self.is_demo:
            loss, accuracy, recall, f1, precision, conf_matrix = test(self.mlp, self.x_test, self.y_test, self.device)

            # Mostrar métricas por consola
            print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Precision: {precision}")
            print("Matriz de confusión:")
            print(conf_matrix)
            print("-" * 60)

            # Devolver métricas al servidor
            metrics = {
                "test_accuracy": accuracy,
                "test_recall": recall,
                "test_f1": f1,
                "test_loss": loss,
                "test_precision": precision,
                "test_conf_matrix": str(conf_matrix) 
            }

        return loss, len(self.x_test), metrics

def client_fn(context: Context):
    # Load model and data
    print("🔍 node_config recibido:", context.node_config)
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
    trainloader, valloader, x_test, y_test, input_size, num_classes = load_data(dataset_path, batch_size, device)
    mlp = MLP(input_size, num_classes)

    return FlowerClient(mlp, trainloader, valloader, x_test, y_test, local_epochs, lr, batch_size, device, wandb, target_delta, noise_multiplier, max_grad_norm, use_dp, use_sa, is_demo, timeout).to_client()

app = ClientApp(
    client_fn=client_fn
)
# app = ClientApp(
#     client_fn=client_fn,
#     mods=[
#         secaggplus_mod,
#     ],
# )
