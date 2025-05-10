#!/bin/bash
# ============================================================
# Script para generar las llaves necesarias para autenticación en Flower
# ============================================================

# script se detenga si ocurre un error en cualquier comando
set -e

# Cambia el directorio
cd "$(dirname "${BASH_SOURCE[0]}")"

# directorio donde se almacenarán las claves generadas.
KEY_DIR=keys

# Crea el directorio si no existe (-p evita errores si ya existe).
mkdir -p "$KEY_DIR"

# Elimina cualquier clave existente en el directorio para evitar conflictos con nuevas claves.
rm -f "$KEY_DIR"/*

# ------------------------------------------------------------
# Función para generar credenciales de cliente
# Argumento:
#   $1 -> Número de clientes a generar (por defecto, 2)
# ------------------------------------------------------------
generate_client_credentials() {
    local num_clients=${1:-2}  # Si no se proporciona un número, usa 2 clientes por defecto.

    # Bucle que genera las claves para cada cliente.
    for ((i=1; i<=num_clients; i++)); do
        # Genera un par de claves ECDSA con 384 bits sin passphrase (-N "")
        # -t ecdsa: Usa el algoritmo de firma de curva elíptica (ECDSA)
        # -b 384: Usa claves de 384 bits (seguridad intermedia)
        # -N "": No se establece contraseña para la clave privada
        # -f: Especifica el archivo de salida para la clave
        ssh-keygen -t ecdsa -b 384 -N "" -f "${KEY_DIR}/client_credentials_$i" -C ""
    done
}

# Llama a la función para generar credenciales, pasando el número de clientes como argumento.
generate_client_credentials "$1"

# ------------------------------------------------------------
# Creación del archivo CSV con las claves públicas de los clientes
# ------------------------------------------------------------

# Se toma la clave pública del primer cliente, se elimina el último carácter y se escribe en el CSV.
# `sed 's/.$//'` elimina el último carácter (suele ser una nueva línea innecesaria).
printf "%s" "$(cat "${KEY_DIR}/client_credentials_1.pub" | sed 's/.$//')" > "$KEY_DIR/client_public_keys.csv"

# Itera sobre los clientes restantes y añade sus claves públicas al archivo CSV, separadas por comas.
for ((i=2; i<=${1:-2}; i++)); do
    printf ",%s" "$(sed 's/.$//' < "${KEY_DIR}/client_credentials_$i.pub")" >> "$KEY_DIR/client_public_keys.csv"
done

# Agrega un salto de línea al final del CSV para garantizar que sea un archivo bien formado.
printf "\n" >> "$KEY_DIR/client_public_keys.csv"

# ------------------------------------------------------------
# Mensaje de finalización
# ------------------------------------------------------------
echo "Generación de claves completada. Claves almacenadas en: $KEY_DIR"
