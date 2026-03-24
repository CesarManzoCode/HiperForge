"""
Utilidad: Hashing y verificación de integridad

Centraliza todas las operaciones de hashing del sistema.
Usado principalmente para:

  1. INTEGRIDAD DE ARCHIVOS:
     FileRef calcula el SHA-256 de un archivo al crearse.
     Antes de procesar un archivo, el agente puede verificar que
     no cambió desde la última vez que lo leyó.

  2. INTEGRIDAD DE DATOS JSON:
     El storage puede calcular un checksum del contenido JSON
     antes de escribirlo y verificarlo al leerlo.
     Detecta corrupción silenciosa en disco.

  3. CACHÉ DE PROMPTS:
     Si el mismo prompt se ejecuta dos veces, el hash del prompt
     permite detectarlo y reutilizar el plan anterior.

¿Por qué SHA-256 y no MD5 o SHA-1?
  MD5 y SHA-1 tienen colisiones conocidas — dos archivos distintos
  pueden producir el mismo hash. SHA-256 no tiene este problema en
  la práctica. Para checksums de integridad, SHA-256 es el estándar.

USO:
  from hiperforge.core.utils.hashing import checksum_file, hash_str, hash_dict

  file_hash = checksum_file(Path("main.py"))     # "a3f2c1..."
  str_hash  = hash_str("contenido del prompt")   # "b4e8d2..."
  dict_hash = hash_dict({"key": "value"})        # "c9f1a3..."
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Algoritmo estándar para todo el sistema.
# Centralizado aquí para que un cambio futuro afecte a todo.
_HASH_ALGORITHM = "sha256"

# Tamaño de chunk para leer archivos grandes sin cargarlos completos en RAM.
# 64 KB es un balance entre número de syscalls y uso de memoria.
_READ_CHUNK_SIZE = 65_536  # 64 KB


def checksum_file(path: Path) -> str:
    """
    Calcula el SHA-256 de un archivo leyéndolo en chunks.

    Lee el archivo en bloques de 64 KB para no cargar archivos
    grandes completos en RAM. Un archivo de 500 MB se procesa
    con el mismo uso de memoria que uno de 1 KB.

    Parámetros:
        path: Ruta al archivo. Debe existir y ser legible.

    Returns:
        String hexadecimal de 64 caracteres con el hash SHA-256.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        PermissionError:   Si el archivo no es legible.
    """
    hasher = hashlib.new(_HASH_ALGORITHM)

    with path.open("rb") as f:
        while chunk := f.read(_READ_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


def checksum_bytes(data: bytes) -> str:
    """
    Calcula el SHA-256 de un bloque de bytes en memoria.

    Usado cuando ya tenemos los bytes cargados (ej: después de
    leer un archivo para procesarlo, calculamos el hash de lo
    que ya leímos sin volver a leer el archivo).

    Parámetros:
        data: Bytes a hashear.

    Returns:
        String hexadecimal de 64 caracteres con el hash SHA-256.
    """
    return hashlib.new(_HASH_ALGORITHM, data).hexdigest()


def hash_str(value: str, *, encoding: str = "utf-8") -> str:
    """
    Calcula el SHA-256 de un string.

    Usado para:
      - Detectar prompts duplicados (cachear planes).
      - Crear identificadores derivados de contenido.
      - Verificar que un string no fue modificado.

    Parámetros:
        value:    String a hashear.
        encoding: Encoding para convertir a bytes. Default UTF-8.

    Returns:
        String hexadecimal de 64 caracteres con el hash SHA-256.
    """
    return checksum_bytes(value.encode(encoding))


def hash_dict(data: dict) -> str:
    """
    Calcula el SHA-256 de un diccionario serializado como JSON.

    La serialización es determinista: claves ordenadas alfabéticamente,
    sin espacios extra. Dos dicts con el mismo contenido siempre
    producen el mismo hash, independientemente del orden de inserción.

    Usado para:
      - Detectar si un JSON en disco fue modificado externamente.
      - Cachear resultados basados en el contenido de los argumentos.

    Parámetros:
        data: Diccionario a hashear. Debe ser serializable a JSON.

    Returns:
        String hexadecimal de 64 caracteres con el hash SHA-256.

    Raises:
        TypeError: Si el dict contiene valores no serializables a JSON.
    """
    # sort_keys=True garantiza orden determinista
    # separators=(',', ':') elimina espacios para compactar
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hash_str(canonical)


def verify_checksum(path: Path, expected_checksum: str) -> bool:
    """
    Verifica que el checksum actual de un archivo coincide con el esperado.

    Más expresivo que comparar checksums manualmente en el caller.

    Parámetros:
        path:              Ruta al archivo a verificar.
        expected_checksum: Checksum SHA-256 esperado (64 chars hex).

    Returns:
        True si el archivo no fue modificado.
        False si el checksum no coincide o el archivo no existe.
    """
    try:
        current = checksum_file(path)
        return current == expected_checksum
    except (FileNotFoundError, PermissionError):
        # Si no podemos leer el archivo, lo consideramos como "no verificado"
        return False


def short_hash(value: str, *, length: int = 8) -> str:
    """
    Devuelve los primeros N caracteres del hash de un string.

    Usado para crear identificadores cortos legibles en logs y CLI
    cuando el hash completo de 64 chars es demasiado largo.

    Ejemplo:
        short_hash("01HX4KM2Q3R5T6V7W8X9Y0Z1A2")  → "a3f2c1b4"

    ATENCIÓN: Los hashes cortos NO son seguros para criptografía.
    Solo usarlos para display e identificación visual, nunca para
    verificación de integridad donde se requiere el hash completo.

    Parámetros:
        value:  String a hashear.
        length: Longitud del hash corto. Default 8 caracteres.

    Returns:
        String hexadecimal de `length` caracteres.
    """
    return hash_str(value)[:length]
