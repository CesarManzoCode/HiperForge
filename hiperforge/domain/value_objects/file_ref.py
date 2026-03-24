"""
Value object: FileRef

Representa una referencia a un archivo en el sistema de archivos.
Es inmutable — captura el estado de un archivo en un momento específico.

¿Para qué sirve?
  Cuando el agente trabaja con archivos (leer, escribir, analizar código),
  necesita más que solo la ruta. Necesita saber:
    - ¿El archivo cambió desde la última vez que lo vimos? (checksum)
    - ¿Qué tipo de archivo es? (mime_type)
    - ¿Cuánto pesa? (size_bytes)

  FileRef captura toda esa información en un snapshot inmutable.
  Si el archivo cambia en disco, el FileRef viejo sigue siendo válido
  como registro histórico de cómo era el archivo en ese momento.

USO TÍPICO:
  # Crear desde una ruta en disco (calcula checksum y mime automáticamente)
  ref = FileRef.from_path(Path("/proyecto/main.py"))

  # Verificar si el archivo cambió desde que lo capturamos
  if ref.has_changed():
      print("El archivo fue modificado externamente")

  # Guardar en JSON junto con una Task
  data = ref.to_dict()
  ref = FileRef.from_dict(data)

  # Info rápida para logs
  print(ref)  # main.py (text/x-python, 2.3 KB)
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tipos MIME conocidos que el agente puede procesar activamente.
# Para otros tipos, FileRef es válido pero el agente solo puede leerlos
# como bytes, no analizarlos con herramientas de código.
# ---------------------------------------------------------------------------

PROCESSABLE_MIME_TYPES: frozenset[str] = frozenset({
    # Código fuente
    "text/x-python",
    "text/javascript",
    "text/typescript",
    "text/x-go",
    "text/x-rust",
    "text/x-java",
    "text/x-c",
    "text/x-c++",
    # Configuración y datos
    "text/plain",
    "application/json",
    "application/x-yaml",
    "text/x-toml",
    "text/html",
    "text/css",
    "text/x-shellscript",
    # Documentación
    "text/markdown",
    "text/x-rst",
})


@dataclass(frozen=True)
class FileRef:
    """
    Referencia inmutable a un archivo en el sistema de archivos.

    Atributos:
        path:       Ruta absoluta al archivo.
        checksum:   Hash SHA-256 del contenido en el momento de creación.
                    Permite detectar si el archivo cambió. None si no se calculó.
        mime_type:  Tipo MIME detectado. None si no se pudo determinar.
        size_bytes: Tamaño del archivo en bytes. None si no se midió.
    """

    path: Path
    checksum: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None

    # ------------------------------------------------------------------
    # Constructor inteligente — crea un FileRef leyendo el archivo real
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path, *, compute_checksum: bool = True) -> FileRef:
        """
        Crea un FileRef desde una ruta real en disco.

        Lee el archivo para calcular su checksum y tamaño.
        Detecta el tipo MIME automáticamente desde la extensión.

        Parámetros:
            path:             Ruta al archivo. Se convierte a absoluta automáticamente.
            compute_checksum: Si False, omite el cálculo del hash (más rápido
                              para archivos grandes que solo necesitamos referenciar).

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta indicada.
        """
        # Normalizamos siempre a ruta absoluta para evitar ambigüedades
        absolute_path = path.resolve()

        if not absolute_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {absolute_path}")

        if not absolute_path.is_file():
            raise ValueError(f"La ruta no es un archivo: {absolute_path}")

        # Detectar MIME desde la extensión del archivo
        mime_type, _ = mimetypes.guess_type(str(absolute_path))

        # Leer el archivo una sola vez para checksum y tamaño
        raw_bytes = absolute_path.read_bytes()
        size_bytes = len(raw_bytes)

        checksum = None
        if compute_checksum:
            checksum = hashlib.sha256(raw_bytes).hexdigest()

        return cls(
            path=absolute_path,
            checksum=checksum,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Nombre del archivo sin la ruta. Ej: 'main.py'"""
        return self.path.name

    @property
    def extension(self) -> str:
        """
        Extensión del archivo en minúsculas con el punto. Ej: '.py'
        Devuelve string vacío si el archivo no tiene extensión.
        """
        return self.path.suffix.lower()

    @property
    def is_processable(self) -> bool:
        """
        True si el agente puede analizar activamente el contenido del archivo.

        Un archivo no procesable (ej: .png, .exe) puede ser referenciado
        pero no analizado con herramientas de código.
        """
        if self.mime_type is None:
            return False
        return self.mime_type in PROCESSABLE_MIME_TYPES

    @property
    def size_kb(self) -> float | None:
        """Tamaño en kilobytes, redondeado a 1 decimal. None si no se midió."""
        if self.size_bytes is None:
            return None
        return round(self.size_bytes / 1024, 1)

    def exists(self) -> bool:
        """
        Verifica si el archivo todavía existe en disco.

        Nota: esto hace I/O — no llamar en bucles tight.
        """
        return self.path.exists() and self.path.is_file()

    def has_changed(self) -> bool:
        """
        Verifica si el archivo fue modificado desde que se creó este FileRef.

        Recalcula el SHA-256 del archivo actual y lo compara con el guardado.
        Devuelve False si no tenemos checksum original (no podemos comparar).

        Nota: esto lee el archivo completo — no llamar frecuentemente en
        archivos grandes.
        """
        # Sin checksum original no podemos detectar cambios
        if self.checksum is None:
            return False

        if not self.exists():
            # Si ya no existe, definitivamente "cambió"
            return True

        current_checksum = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return current_checksum != self.checksum

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa para guardar en JSON.

        La ruta se guarda como string porque Path no es serializable por json.
        """
        return {
            "path": str(self.path),
            "checksum": self.checksum,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileRef:
        """
        Reconstruye un FileRef desde un diccionario leído del JSON.

        No verifica si el archivo existe — es una referencia histórica
        y el archivo puede haber sido movido o eliminado desde entonces.
        """
        return cls(
            path=Path(data["path"]),
            checksum=data.get("checksum"),
            mime_type=data.get("mime_type"),
            size_bytes=data.get("size_bytes"),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo: main.py (text/x-python, 2.3 KB)
        """
        mime_label = self.mime_type or "tipo desconocido"
        size_label = f"{self.size_kb} KB" if self.size_kb is not None else "tamaño desconocido"
        return f"{self.name} ({mime_label}, {size_label})"
