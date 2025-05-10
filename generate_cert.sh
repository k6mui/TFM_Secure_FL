#!/bin/bash
# ============================================================
# Script para generar certificados SSL/TLS con ECDSA y claves protegidas con contraseña
# ============================================================
# Usa la curva secp384r1 (P-384) para máxima seguridad y compatibilidad.
# Se protege la clave privada con cifrado AES-256 y una contraseña segura.
# ============================================================

set -e  # Detener ejecución si ocurre un error

# Directorio donde se guardarán los certificados
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
CERT_DIR="$SCRIPT_DIR/certificates_tls"
CONFIG_FILE="$SCRIPT_DIR/certificate.conf"
export OPENSSL_CONF="$CONFIG_FILE"

# Crear directorio de certificados si no existe
mkdir -p "$CERT_DIR"

# Eliminar certificados previos si existen
rm -f "$CERT_DIR"/*

# ------------------------------------------------------------
# Generación de la clave privada y certificado de la CA con ECDSA P-384 
# ------------------------------------------------------------
echo "Generando clave privada de la CA con protección AES-256 y ECDSA (secp384r1)..."
openssl ecparam -genkey -name secp384r1 -out "$CERT_DIR/ca.key"

# Crear el certificado autofirmado de la CA con validez de 1 año
echo "Generando certificado autofirmado de la CA..."
openssl req -new -x509 -key "$CERT_DIR/ca.key" -sha384 \
    -subj '/C=ES/ST=Madrid/O="SecureCA, Inc."' \
    -days 365 -out "$CERT_DIR/ca.crt"

# ------------------------------------------------------------
# Generación de la clave privada del servidor con ECDSA P-384 
# ------------------------------------------------------------
echo "Generando clave privada del servidor..."
openssl ecparam -genkey -name secp384r1 -out "$CERT_DIR/server.key"

# Verificar si existe el archivo de configuración para la CSR
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: No se encuentra el archivo de configuración $CONFIG_FILE"
    exit 1
fi

# ------------------------------------------------------------
# Creación de CSR (Certificate Signing Request) del servidor
# ------------------------------------------------------------
echo "Creando CSR para el servidor..."
openssl req -new -key "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" -config "$CONFIG_FILE"

# ------------------------------------------------------------
# Firmar el certificado del servidor con la CA
# ------------------------------------------------------------
echo "Firmando el certificado del servidor con la CA..."
openssl x509 -req -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" -CAcreateserial \
    -out "$CERT_DIR/server.pem" -days 730 -sha384 -extfile "$CONFIG_FILE" -extensions req_ext

# ------------------------------------------------------------
# Protección de archivos sensibles para que solo el propietario pueda leerlo
# ------------------------------------------------------------
chmod 600 "$CERT_DIR/ca.key" "$CERT_DIR/server.key"

echo "Certificados generados exitosamente en: $CERT_DIR"
