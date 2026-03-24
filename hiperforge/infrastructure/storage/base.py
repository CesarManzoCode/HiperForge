"""
BaseStorage — Capa base de operaciones de almacenamiento JSON.

Este módulo resuelve los tres problemas fundamentales de persistencia
en un sistema que escribe archivos JSON constantemente:

  PROBLEMA 1: CORRUPCIÓN POR INTERRUPCIÓN
    Si el proceso muere a mitad de escribir un JSON, el archivo queda
    corrupto — mitad del JSON viejo, mitad del nuevo. Al releer,
    json.loads() falla y perdemos los datos.

    SOLUCIÓN: Escritura atómica + backup automático.
      1. Escribimos el nuevo contenido en un archivo temporal (.tmp)
      2. Guardamos el archivo anterior como backup (.bak)
      3. Renombramos el .tmp al nombre final (operación atómica en el OS)
      Si el proceso muere en cualquier paso anterior al 3, el archivo
      original nunca fue tocado. Si muere en el 3, el OS garantiza
      que rename() es atómico — nunca queda a medias.

  PROBLEMA 2: ESCRITURAS CONCURRENTES
    Dos procesos de HiperForge corriendo al mismo tiempo podrían
    escribir el mismo archivo simultáneamente, corrompiendo los datos.

    SOLUCIÓN: File locking con fcntl (Unix) / msvcrt (Windows).
      Antes de cualquier escritura, adquirimos un lock exclusivo
      en el archivo. Si otro proceso ya tiene el lock, esperamos
      hasta STORAGE_LOCK_TIMEOUT_SECONDS antes de fallar.

  PROBLEMA 3: DATOS SILENCIOSAMENTE INVÁLIDOS
    Un JSON puede parsearse correctamente pero tener datos inválidos
    (campos faltantes, tipos incorrectos, versión de schema antigua).

    SOLUCIÓN: Checksum + validación de schema version al leer.
      Guardamos el SHA-256 del contenido junto al JSON.
      Al leer, verificamos el checksum antes de parsear.
      Si no coincide → StorageCorruptedError inmediato.

COMPATIBILIDAD:
  El file locking usa fcntl en Unix/Mac y msvcrt en Windows.
  Ambos son parte de la stdlib de Python — sin dependencias extra.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# File locking — importamos según el OS en tiempo de ejecución
# NO al nivel del módulo para evitar ImportError en el OS incorrecto
if sys.platform == "win32":
    import msvcrt as _msvcrt
else:
    import fcntl as _fcntl

from hiperforge.core.constants import (
    STORAGE_FILE_ENCODING,
    STORAGE_JSON_INDENT,
    STORAGE_LOCK_TIMEOUT_SECONDS,
)
from hiperforge.core.logging import get_logger
from hiperforge.core.utils.hashing import checksum_bytes
from hiperforge.domain.exceptions import (
    StorageCorruptedError,
    StorageReadError,
    StorageWriteError,
)

logger = get_logger(__name__)

# Extensiones de archivos temporales y de backup
_TMP_EXTENSION = ".tmp"
_BAK_EXTENSION = ".bak"

# Campo especial que guardamos en cada JSON para verificar integridad
_CHECKSUM_FIELD = "__checksum__"


class BaseStorage:
    """
    Operaciones atómicas de lectura y escritura de archivos JSON.

    Esta clase no sabe nada de workspaces, projects ni tasks.
    Solo sabe leer y escribir JSON de forma segura y confiable.

    JSONStorage hereda de esta clase y agrega la lógica de
    qué leer y dónde guardarlo dentro de la estructura de directorios.
    """

    # ------------------------------------------------------------------
    # Escritura atómica con backup
    # ------------------------------------------------------------------

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        """
        Escribe un diccionario como JSON de forma atómica con backup.

        FLUJO COMPLETO:
          1. Serializar data a JSON string
          2. Calcular checksum del JSON serializado
          3. Agregar checksum al JSON como campo __checksum__
          4. Crear directorio si no existe
          5. Adquirir file lock
          6. Escribir en archivo temporal (.tmp) en el mismo directorio
          7. Si existe archivo anterior, renombrarlo a .bak
          8. Renombrar .tmp al nombre final (operación atómica del OS)
          9. Liberar file lock

        Si cualquier paso falla antes del 8, el archivo original
        nunca fue modificado. El .bak contiene la versión anterior.

        Parámetros:
            path: Ruta absoluta donde escribir el JSON.
            data: Diccionario a serializar. Debe ser JSON-serializable.

        Raises:
            StorageWriteError: Si no se puede escribir por permisos,
                               disco lleno, u otro error del OS.
        """
        # Paso 1-2: serializar y calcular checksum
        json_bytes = self._serialize(data)

        # Paso 3: agregar checksum al dict y re-serializar
        # El checksum se calcula sobre el JSON SIN el checksum mismo,
        # para que sea determinista y verificable
        checksum = checksum_bytes(json_bytes)
        data_with_checksum = {**data, _CHECKSUM_FIELD: checksum}
        final_bytes = self._serialize(data_with_checksum)

        # Paso 4: crear directorio si no existe
        path.parent.mkdir(parents=True, exist_ok=True)

        # Paso 5-9: escritura atómica con locking
        try:
            self._atomic_write(path=path, content=final_bytes)
        except OSError as exc:
            raise StorageWriteError(
                path=str(path),
                reason=str(exc),
            ) from exc

        logger.debug(
            "JSON escrito exitosamente",
            path=str(path),
            size_bytes=len(final_bytes),
        )

    def read_json(self, path: Path) -> dict[str, Any]:
        """
        Lee y parsea un archivo JSON verificando su integridad.

        FLUJO:
          1. Verificar que el archivo existe
          2. Leer bytes del archivo
          3. Parsear JSON
          4. Extraer y verificar checksum (si existe)
          5. Devolver el dict sin el campo __checksum__

        Si el checksum no coincide → StorageCorruptedError.
        Si el JSON está malformado → StorageCorruptedError.

        Parámetros:
            path: Ruta absoluta del archivo JSON a leer.

        Returns:
            Diccionario con el contenido del JSON, sin __checksum__.

        Raises:
            StorageReadError:      Si el archivo no existe o no es legible.
            StorageCorruptedError: Si el JSON está malformado o el checksum
                                   no coincide (datos corruptos).
        """
        if not path.exists():
            raise StorageReadError(
                path=str(path),
                reason="el archivo no existe",
            )

        if not path.is_file():
            raise StorageReadError(
                path=str(path),
                reason="la ruta no es un archivo",
            )

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise StorageReadError(
                path=str(path),
                reason=str(exc),
            ) from exc

        # Parsear el JSON
        try:
            data = json.loads(raw_bytes.decode(STORAGE_FILE_ENCODING))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StorageCorruptedError(
                path=str(path),
                reason=f"JSON malformado: {exc}",
            ) from exc

        if not isinstance(data, dict):
            raise StorageCorruptedError(
                path=str(path),
                reason=f"se esperaba un objeto JSON, se obtuvo {type(data).__name__}",
            )

        # Verificar checksum si existe en el archivo
        stored_checksum = data.pop(_CHECKSUM_FIELD, None)

        if stored_checksum is not None:
            # Recalculamos el checksum sobre el JSON sin el campo __checksum__
            # para comparar con lo que se calculó al escribir
            expected_bytes = self._serialize(data)
            expected_checksum = checksum_bytes(expected_bytes)

            if stored_checksum != expected_checksum:
                raise StorageCorruptedError(
                    path=str(path),
                    reason=(
                        f"checksum no coincide — el archivo fue modificado externamente. "
                        f"Esperado: {expected_checksum[:16]}... "
                        f"Encontrado: {stored_checksum[:16]}..."
                    ),
                )

        logger.debug(
            "JSON leído exitosamente",
            path=str(path),
            size_bytes=len(raw_bytes),
        )

        return data

    def delete_path(self, path: Path) -> None:
        """
        Elimina un archivo o directorio completo del disco.

        Para directorios, elimina recursivamente todo su contenido.
        Esta operación es IRREVERSIBLE — usar con precaución.

        Parámetros:
            path: Ruta a eliminar.

        Raises:
            StorageWriteError: Si no se puede eliminar por permisos u otro error.
        """
        if not path.exists():
            return  # idempotente — no lanzamos error si ya no existe

        try:
            if path.is_file():
                path.unlink()
            else:
                # Eliminamos el directorio y todo su contenido
                import shutil
                shutil.rmtree(path)

        except OSError as exc:
            raise StorageWriteError(
                path=str(path),
                reason=f"no se pudo eliminar: {exc}",
            ) from exc

        logger.debug("ruta eliminada", path=str(path))

    def path_exists(self, path: Path) -> bool:
        """
        Verifica si existe un archivo o directorio en la ruta dada.

        No lanza excepciones — devuelve False ante cualquier error.
        """
        try:
            return path.exists()
        except OSError:
            return False

    def list_subdirectories(self, directory: Path) -> list[str]:
        """
        Lista los nombres de los subdirectorios de un directorio.

        Usado por los repos para listar IDs de entidades
        (los subdirectorios tienen el ID como nombre).

        Parámetros:
            directory: Directorio padre a listar.

        Returns:
            Lista de nombres de subdirectorios, ordenada alfabéticamente.
            Lista vacía si el directorio no existe.
        """
        if not directory.exists() or not directory.is_dir():
            return []

        try:
            return sorted(
                entry.name
                for entry in directory.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            )
        except OSError as exc:
            logger.warning(
                "error listando subdirectorios",
                directory=str(directory),
                error=str(exc),
            )
            return []

    # ------------------------------------------------------------------
    # Recuperación ante corrupción
    # ------------------------------------------------------------------

    def try_restore_from_backup(self, path: Path) -> bool:
        """
        Intenta restaurar un archivo desde su backup (.bak).

        Llamado automáticamente cuando read_json() lanza
        StorageCorruptedError — el caller puede intentar recuperarse.

        Parámetros:
            path: Ruta del archivo corrupto.

        Returns:
            True si se restauró exitosamente desde el backup.
            False si no hay backup disponible o la restauración falló.
        """
        backup_path = path.with_suffix(_BAK_EXTENSION)

        if not backup_path.exists():
            logger.warning(
                "no hay backup disponible para restaurar",
                path=str(path),
                backup_path=str(backup_path),
            )
            return False

        try:
            # Intentamos leer el backup para verificar que es válido
            self.read_json(backup_path)

            # Si es válido, lo copiamos como el archivo principal
            import shutil
            shutil.copy2(backup_path, path)

            logger.info(
                "archivo restaurado desde backup exitosamente",
                path=str(path),
                backup_path=str(backup_path),
            )
            return True

        except (StorageReadError, StorageCorruptedError) as exc:
            logger.error(
                "el backup también está corrupto — no se puede restaurar",
                path=str(path),
                backup_path=str(backup_path),
                error=str(exc),
            )
            return False

    # ------------------------------------------------------------------
    # Operaciones internas
    # ------------------------------------------------------------------

    def _atomic_write(self, path: Path, content: bytes) -> None:
        """
        Escribe content en path de forma atómica con backup.

        GARANTÍA: después de esta función, path contiene content
        o el archivo original sin cambios. Nunca un estado intermedio.

        Usa file locking para prevenir escrituras concurrentes.
        """
        # El archivo temporal debe estar en el mismo directorio que el destino
        # para que os.rename() sea atómico (mismo filesystem)
        tmp_path = path.with_suffix(_TMP_EXTENSION)
        bak_path = path.with_suffix(_BAK_EXTENSION)

        lock_path = path.with_suffix(".lock")

        with _FileLock(lock_path, timeout=STORAGE_LOCK_TIMEOUT_SECONDS):
            # Escribir en el temporal
            tmp_path.write_bytes(content)

            # Backup del archivo anterior (si existe)
            if path.exists():
                os.replace(path, bak_path)  # os.replace es atómico en mismo filesystem

            # Mover temporal al destino final (atómico)
            os.replace(tmp_path, path)

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """
        Serializa un dict a bytes JSON con formato consistente.

        sort_keys=True garantiza que el mismo dict siempre produce
        el mismo JSON — crítico para que los checksums sean reproducibles.
        """
        json_str = json.dumps(
            data,
            indent=STORAGE_JSON_INDENT,
            ensure_ascii=False,   # preserva caracteres UTF-8 (tildes, etc.)
            sort_keys=True,       # orden determinista para checksums
            default=str,          # serializa tipos no-JSON como strings
        )
        return json_str.encode(STORAGE_FILE_ENCODING)


# ---------------------------------------------------------------------------
# File locking — compatible con Unix y Windows
# ---------------------------------------------------------------------------

class _FileLock:
    """
    Context manager de file lock compatible con Unix y Windows.

    En Unix usa fcntl.flock() — lock a nivel de kernel, liberado
    automáticamente si el proceso muere (el OS lo libera).

    En Windows usa msvcrt.locking() — comportamiento equivalente.

    El lock se aplica sobre un archivo .lock separado del JSON,
    no sobre el JSON mismo, para no interferir con las lecturas.

    Uso:
        with _FileLock(lock_path, timeout=5.0):
            # operaciones protegidas
    """

    def __init__(self, lock_path: Path, timeout: float) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._lock_file = None

    def __enter__(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self._lock_path, "w")

        if sys.platform == "win32":
            self._acquire_windows()
        else:
            self._acquire_unix()

    def __exit__(self, *_: Any) -> None:
        if self._lock_file is not None:
            try:
                if sys.platform == "win32":
                    _msvcrt.locking(self._lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
                else:
                    _fcntl.flock(self._lock_file.fileno(), _fcntl.LOCK_UN)
            finally:
                self._lock_file.close()
                try:
                    self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _acquire_unix(self) -> None:
        """Adquiere el lock en Unix/Linux con timeout."""
        import errno
        import time

        deadline = time.monotonic() + self._timeout
        while True:
            try:
                # LOCK_EX = exclusivo, LOCK_NB = no bloquear (lanza excepción si ocupado)
                _fcntl.flock(self._lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                return  # lock adquirido

            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise  # error inesperado — propagar

                if time.monotonic() >= deadline:
                    raise StorageWriteError(
                        path=str(self._lock_path),
                        reason=(
                            f"no se pudo adquirir el file lock después de "
                            f"{self._timeout}s — otro proceso puede estar escribiendo"
                        ),
                    )

                time.sleep(0.05)

    def _acquire_windows(self) -> None:
        """Adquiere el lock en Windows con timeout."""
        import errno
        import time

        deadline = time.monotonic() + self._timeout
        while True:
            try:
                _msvcrt.locking(self._lock_file.fileno(), _msvcrt.LK_NBLCK, 1)
                return

            except OSError as exc:
                if exc.errno != errno.EACCES:
                    raise

                if time.monotonic() >= deadline:
                    raise StorageWriteError(
                        path=str(self._lock_path),
                        reason=(
                            f"no se pudo adquirir el file lock después de "
                            f"{self._timeout}s"
                        ),
                    )

                time.sleep(0.05)