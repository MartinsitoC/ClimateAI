#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========================================================"
echo "Iniciando URBAN-HEAT GRID INTELLIGENCE Dashboard..."
echo "Presiona Ctrl + C en esta ventana para detener el servidor."
echo "========================================================"
python3 main.py
