
# 🚀 Flower CLI Deployment Guide for Federated Learning

This guide serves as a practical reference to deploy a federated learning system using Flower, including how to launch the `SuperLink`, `SuperNodes`, and run `ClientApp/ServerApp` components. It also covers options for secure communication via TLS and REST.

---

## 🔧 Activating the Virtual Environment

### Linux / macOS
```bash
source flwrEnv/bin/activate && cd tfmapp
```

### PowerShell (Windows)
```powershell
./flwrEnv/Scripts/activate && cd tfmapp
```

## 🌐 Starting the SuperLink

### gRPC + TLS + Public Key Authentication

#### Linux / macOS
```bash
FLWR_LOG_LEVEL=DEBUG flower-superlink     --fleet-api-address 127.0.0.1:9092     --ssl-ca-certfile certificates_tls/ca.crt     --ssl-certfile certificates_tls/server.pem     --ssl-keyfile certificates_tls/server.key     --auth-list-public-keys keys/client_public_keys.csv
```

#### PowerShell (Windows)
```powershell
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-superlink `
    --fleet-api-address 127.0.0.1:9092 `
    --ssl-ca-certfile certificates_tls/ca.crt `
    --ssl-certfile certificates_tls/server.pem `
    --ssl-keyfile certificates_tls/server.key `
    --auth-list-public-keys keys/client_public_keys.csv
```

---

### REST + TLS 

```bash
FLWR_LOG_LEVEL=DEBUG flower-superlink     --fleet-api-type=rest     --ssl-ca-certfile certificates_tls/ca.crt     --ssl-certfile certificates_tls/server.pem     --ssl-keyfile certificates_tls/server.key
```

---

## 🧩 Starting SuperNodes (Clients)

### Tabular CSV Clients

```bash
flower-supernode --root-certificates ./certificates_tls/ca.crt     --superlink 127.0.0.1:9092     --clientappio-api-address 127.0.0.1:9094     --node-config 'dataset-path="datasets/tabular/balancedFederatedDatasets/balanced_client1.csv"'     --auth-supernode-private-key keys/client_credentials_1     --auth-supernode-public-key keys/client_credentials_1.pub
```

Repeat for other clients by changing ports and keys:

```bash
# Client 1
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "dataset-path='datasets/tabular/federatedDatasets/smoking_client1.csv'" --auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub

# Client 2
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config "dataset-path='datasets/tabular/federatedDatasets/smoking_client2.csv'" --auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub

# Client 3
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config "dataset-path='datasets/tabular/federatedDatasets/smoking_client3.csv'" --auth-supernode-private-key keys/client_credentials_3 --auth-supernode-public-key keys/client_credentials_3.pub

# Client 4
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9097 --node-config "dataset-path='datasets/tabular/federatedDatasets/smoking_client4.csv'" --auth-supernode-private-key keys/client_credentials_4 --auth-supernode-public-key keys/client_credentials_4.pub
```

### Images CSV Clients
```bash
# Client 1 (Image)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "dataset-path='datasets/imgs/federatedDatasets/client_1'" --auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub

# Client 2 (Image)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config "dataset-path='datasets/imgs/federatedDatasets/client_2'" --auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub

# Client 3 (Image)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config "dataset-path='datasets/imgs/federatedDatasets/client_3'" --auth-supernode-private-key keys/client_credentials_3 --auth-supernode-public-key keys/client_credentials_3.pub

# Client 4 (Image)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9097 --node-config "dataset-path='datasets/imgs/federatedDatasets/client_4'" --auth-supernode-private-key keys/client_credentials_4 --auth-supernode-public-key keys/client_credentials_4.pub
```

### Text CSV Clients
```bash
# Client 1 (NLP)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "partition-id=0 num-partitions=4" --auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub

# Client 2 (NLP)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config "partition-id=1 num-partitions=4" --auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub

# Client 3 (NLP)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config "partition-id=2 num-partitions=4" --auth-supernode-private-key keys/client_credentials_3 --auth-supernode-public-key keys/client_credentials_3.pub

# Client 4 (NLP)
$env:FLWR_LOG_LEVEL = "DEBUG"; flower-supernode --root-certificates ./certificates_tls/ca.crt --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9097 --node-config "partition-id=3 num-partitions=4" --auth-supernode-private-key keys/client_credentials_4 --auth-supernode-public-key keys/client_credentials_4.pub
```
---

### (REST)

```bash
flower-supernode     --root-certificates ./certificates_tls/ca.crt     --superlink="https://127.0.0.1:9095"     --rest     --clientappio-api-address 127.0.0.1:9094     --node-config 'dataset-path="datasets/cifar10_part_1"'
```

```bash
flower-supernode     --root-certificates ./certificates_tls/ca.crt     --superlink="https://127.0.0.1:9095"     --rest     --clientappio-api-address 127.0.0.1:9096     --node-config 'dataset-path="datasets/cifar10_part_2"'
```

---

## 🧠 Running the ServerApp

Make sure your `pyproject.toml` is properly configured with your federation:

```toml
[tool.flwr.federations.my_federation]
address = "127.0.0.1:9093"
insecure = true  # Or use:
# root-certificates = "./certificates/ca.crt"
```

### Running ServerApp
```bash
flwr run . my_federation --stream
```

---

---

## 🔍 Inspecting Traffic (TLS Debugging)

### Using `tcpdump`
```bash
sudo tcpdump -i lo port 9093 -vv -X &> logConnection.txt
```

### Recommended
Use **Wireshark** to verify that communication is encrypted.

---


## 📌 Additional Notes

- All nodes must trust the same certificate authority (`ca.crt`).
- Use `--rest` in the SuperNodes if your SuperLink is launched with `--fleet-api-type=rest`.
- Set `FLWR_LOG_LEVEL=DEBUG` for detailed debugging output.

---

