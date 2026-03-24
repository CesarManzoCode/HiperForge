"""
Value object: Message

Representa un mensaje en una conversación con el LLM.
Es inmutable — una vez creado no se modifica, se crea uno nuevo.

¿Por qué inmutable?
  Los mensajes son hechos históricos. El mensaje que el usuario envió
  a las 3pm no "cambia" — si hay una corrección, es un mensaje nuevo.
  La inmutabilidad garantiza que el historial de conversación sea confiable.

USO TÍPICO:
  # Crear mensajes
  user_msg = Message.user("Crea un servidor HTTP en Python")
  assistant_msg = Message.assistant("Aquí está el código...")
  system_msg = Message.system("Eres un agente de desarrollo...")

  # Acceder a los datos
  print(user_msg.role)     # "user"
  print(user_msg.content)  # "Crea un servidor HTTP en Python"

  # Serializar para enviar al LLM
  payload = user_msg.to_dict()
  # {"role": "user", "content": "Crea un servidor HTTP en Python"}

  # Reconstruir desde JSON guardado en disco
  msg = Message.from_dict({"role": "user", "content": "..."})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Role(str, Enum):
    """
    Roles válidos en una conversación con el LLM.

    Hereda de str para que la comparación con strings funcione directamente:
      role == "user"  →  True  (sin necesidad de role == Role.USER)

    Esto es útil al deserializar desde JSON donde los valores vienen como strings.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)  # frozen=True hace el objeto completamente inmutable
class Message:
    """
    Mensaje individual en una conversación con el LLM.

    Atributos:
        role:       Quién envió el mensaje (system, user, assistant).
        content:    Texto del mensaje.
        created_at: Momento exacto de creación (siempre en UTC).
        metadata:   Datos extra opcionales (ej: tokens usados, tool_call_id).
                    Es un dict pero lo guardamos como tuple de pares para
                    mantener la inmutabilidad del dataclass.

    Nota sobre metadata:
        Se expone como dict en la propiedad `meta` pero internamente
        se almacena como tuple de pares (key, value) para respetar
        la restricción frozen=True del dataclass.
    """

    role: Role
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Tupla de pares en vez de dict para poder usar frozen=True
    _metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple, repr=False)

    # ------------------------------------------------------------------
    # Constructores semánticos — más claro que Message(role=Role.USER, ...)
    # ------------------------------------------------------------------

    @classmethod
    def system(cls, content: str, **metadata: Any) -> Message:
        """Crea un mensaje de sistema (instrucciones al LLM)."""
        return cls(
            role=Role.SYSTEM,
            content=content,
            _metadata=tuple(metadata.items()),
        )

    @classmethod
    def user(cls, content: str, **metadata: Any) -> Message:
        """Crea un mensaje del usuario."""
        return cls(
            role=Role.USER,
            content=content,
            _metadata=tuple(metadata.items()),
        )

    @classmethod
    def assistant(cls, content: str, **metadata: Any) -> Message:
        """Crea un mensaje del asistente (respuesta del LLM)."""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            _metadata=tuple(metadata.items()),
        )

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def meta(self) -> dict[str, Any]:
        """Devuelve la metadata como dict normal para fácil acceso."""
        return dict(self._metadata)

    @property
    def is_empty(self) -> bool:
        """True si el contenido está vacío o solo tiene espacios."""
        return not self.content.strip()

    @property
    def word_count(self) -> int:
        """Cantidad aproximada de palabras (útil para estimar tokens)."""
        return len(self.content.split())

    # ------------------------------------------------------------------
    # Serialización — para guardar en JSON y enviar al LLM
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa el mensaje al formato que espera la API del LLM.

        El formato {"role": ..., "content": ...} es estándar entre
        Anthropic, OpenAI y Ollama — los adapters pueden usarlo directamente.
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Serializa el mensaje con TODOS los campos para guardarlo en JSON.

        A diferencia de to_dict(), incluye created_at y metadata
        para poder reconstruir el mensaje exacto desde disco.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """
        Reconstruye un Message desde un diccionario (leído del JSON en disco).

        Maneja dos formatos:
          - Formato completo (guardado por to_storage_dict): tiene created_at y metadata.
          - Formato mínimo (enviado por el LLM): solo role y content.
        """
        # Parseamos created_at si existe, sino usamos el momento actual
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc)
        )

        metadata = data.get("metadata", {})

        return cls(
            role=Role(data["role"]),
            content=data["content"],
            created_at=created_at,
            _metadata=tuple(metadata.items()),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Representación corta para logs y terminal.

        Ejemplo: [user] Crea un servidor HTTP en Python (12 palabras)
        """
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"[{self.role.value}] {preview} ({self.word_count} palabras)"