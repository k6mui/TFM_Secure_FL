import torch
from flwr.server import Grid, LegacyContext, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
from logging import DEBUG
from flwr.common.logger import update_console_handler
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from tfmapp.images_federation.utils.workflow_with_log import SecAggPlusWorkflowWithLogs
from tfmapp.images_federation.utils.task_images import get_weights, get_model
from flwr.common import Context
from tfmapp.images_federation.strategy import CustomFedAvg


# Federated evaluation metrics aggregation function
def weighted_average(metrics):
    accuracies = [m["test_accuracy"] for _, m in metrics]
    recalls = [m["test_recall"] for _, m in metrics]
    precisions = [m["test_precision"] for _, m in metrics]
    f1s = [m["test_f1"] for _, m in metrics]

    return {
        "federated_evaluate_accuracy": sum(accuracies) / len(accuracies),
        "federated_evaluate_recall": sum(recalls) / len(recalls),
        "federated_evaluate_precision": sum(precisions) / len(precisions),
        "federated_evaluate_f1score": sum(f1s) / len(f1s),
    }


# Define the Flower server function
def server_fn(context: Context):
    # Retrieve values from run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    min_clients = context.run_config["min-clients"]
    use_wandb = context.run_config["use-wandb"]

    # Define Flower strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=use_wandb,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Register Flower ServerApp
# app = ServerApp(server_fn=server_fn)

# Flower ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    use_sa = context.run_config.get("use-sa", False)
    is_demo = context.run_config.get("is-demo", False)
    num_rounds = context.run_config["num-server-rounds"]

    if use_sa:
        # Secure Aggregation setup
        strategy = CustomFedAvg(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=context.run_config["fraction-evaluate"],
            min_available_clients=context.run_config["min-clients"],
            min_fit_clients=context.run_config["min-clients"],
            min_evaluate_clients=context.run_config["min-clients"],
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        legacy_context = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        if is_demo:
            
            update_console_handler(DEBUG, True, True)
            fit_workflow = SecAggPlusWorkflowWithLogs(
                num_shares=context.run_config["num-shares"],
                reconstruction_threshold=context.run_config["reconstruction-threshold"],
                max_weight=1,
                timeout=context.run_config["timeout"],
            )
        else:
            fit_workflow = SecAggPlusWorkflow(
                num_shares=context.run_config["num-shares"],
                reconstruction_threshold=context.run_config["reconstruction-threshold"],
                max_weight=context.run_config["max-weight"],
            )

        workflow = DefaultWorkflow(fit_workflow=fit_workflow)
        workflow(grid, legacy_context)

    else:
        # Normal (no secure aggregation)
        strategy = CustomFedAvg(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=context.run_config["fraction-evaluate"],
            min_available_clients=context.run_config["min-clients"],
            min_fit_clients=context.run_config["min-clients"],
            min_evaluate_clients=context.run_config["min-clients"],
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        config = ServerConfig(num_rounds=num_rounds)
        legacy_context = LegacyContext(
            context=context,
            config=config,
            strategy=strategy,
        )

        workflow = DefaultWorkflow()
        workflow(grid, legacy_context)
