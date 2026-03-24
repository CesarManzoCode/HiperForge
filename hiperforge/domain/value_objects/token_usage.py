"""
Value object: TokenUsage

Representa el consumo de tokens y costo estimado de una llamada al LLM.
Es inmutable — registra un hecho histórico que no debe cambiar.

¿Para qué sirve?
  Cada llamada al LLM consume tokens de entrada (el prompt) y de salida
  (la respuesta). Los proveedores cobran por tokens. Este objeto permite:
    - Trackear el costo de cada operación individualmente.
    - Acumular el costo total de una Task o de una sesión completa.
    - Comparar eficiencia entre distintos modelos o prompts.

USO TÍPICO:
  # Crear desde la respuesta del LLM
  usage = TokenUsage(input_tokens=1200, output_tokens=350, model="claude-sonnet-4-6")

  # Ver el costo estimado
  print(usage.total_tokens)       # 1550
  print(usage.estimated_cost_usd) # 0.00432 (depende del modelo)

  # Acumular el costo de varias llamadas
  total = TokenUsage.zero()
  for call in subtask.tool_calls:
      total = total + call.token_usage

  print(total.estimated_cost_usd)  # costo total de la subtask

  # Serializar para guardar en JSON
  data = usage.to_dict()
  usage = TokenUsage.from_dict(data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Precios por millón de tokens (en USD) por modelo.
# Estructura: {model_id: (precio_input, precio_output)}
#
# IMPORTANTE: estos precios cambian con el tiempo.
# Si un modelo no está aquí, el costo estimado será 0.0 con una advertencia.
# Actualizar cuando cambien los precios oficiales de cada proveedor.
# Fuentes:
#   Anthropic → https://www.anthropic.com/pricing
#   OpenAI    → https://openai.com/pricing
# ---------------------------------------------------------------------------

_PRICE_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic — (input_usd, output_usd) por millón de tokens
    "claude-opus-4-6":    (15.00, 75.00),
    "claude-sonnet-4-6":  (3.00,  15.00),
    "claude-haiku-4-5":   (0.80,   4.00),

    # OpenAI
    "gpt-4o":             (5.00,  15.00),
    "gpt-4o-mini":        (0.15,   0.60),
    "gpt-4-turbo":        (10.00, 30.00),
    "o1":                 (15.00, 60.00),

    # Ollama — modelos locales, sin costo monetario
    "llama3":             (0.00,   0.00),
    "mistral":            (0.00,   0.00),
    "codellama":          (0.00,   0.00),
}


@dataclass(frozen=True)
class TokenUsage:
    """
    Consumo de tokens de una llamada al LLM.

    Atributos:
        input_tokens:  Tokens del prompt enviado al modelo.
        output_tokens: Tokens de la respuesta generada por el modelo.
        model:         ID del modelo usado (para calcular el costo).
                       None si el proveedor no informó el modelo.
    """

    input_tokens: int
    output_tokens: int
    model: str | None = None

    # ------------------------------------------------------------------
    # Constructor especial — representa "sin uso" (punto neutro para sumas)
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> TokenUsage:
        """
        Crea un TokenUsage vacío.

        Útil como valor inicial al acumular el costo de múltiples llamadas:
          total = TokenUsage.zero()
          for call in calls:
              total = total + call.usage
        """
        return cls(input_tokens=0, output_tokens=0)

    # ------------------------------------------------------------------
    # Propiedades calculadas
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        """Suma de tokens de entrada y salida."""
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """
        Costo estimado en USD basado en el modelo y los precios conocidos.

        Devuelve 0.0 si:
          - El modelo es None (no informado por el proveedor).
          - El modelo no está en la tabla de precios (ej: modelo nuevo o local).

        El cálculo usa la tabla _PRICE_PER_MILLION_TOKENS definida arriba.
        """
        if self.model is None:
            return 0.0

        prices = _PRICE_PER_MILLION_TOKENS.get(self.model)

        if prices is None:
            # Modelo desconocido — no podemos calcular el costo
            return 0.0

        input_price_per_million, output_price_per_million = prices

        input_cost = (self.input_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.output_tokens / 1_000_000) * output_price_per_million

        return round(input_cost + output_cost, 8)

    @property
    def is_empty(self) -> bool:
        """True si no se consumió ningún token (equivalente a TokenUsage.zero())."""
        return self.total_tokens == 0

    # ------------------------------------------------------------------
    # Operación de suma — para acumular uso de múltiples llamadas
    # ------------------------------------------------------------------

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """
        Suma dos TokenUsage para obtener el consumo total acumulado.

        El modelo del resultado es el de self, a menos que self no tenga modelo,
        en cuyo caso se usa el de other. Esto permite acumular llamadas del
        mismo modelo correctamente.

        Ejemplo:
          total = TokenUsage.zero() + call_1.usage + call_2.usage + call_3.usage
        """
        if not isinstance(other, TokenUsage):
            return NotImplemented

        # Usamos el modelo que esté disponible para estimar el costo acumulado
        model = self.model or other.model

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model=model,
        )

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardar en JSON."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,           # guardamos total para consultas rápidas
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """
        Reconstruye un TokenUsage desde un diccionario leído del JSON.

        Ignora 'total_tokens' y 'estimated_cost_usd' del dict porque
        son valores calculados — siempre se recalculan desde input/output.
        """
        return cls(
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            model=data.get("model"),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo: 1200 in + 350 out = 1550 tokens ($0.00432) [claude-sonnet-4-6]
        """
        model_label = f" [{self.model}]" if self.model else ""
        cost_label = f" (${self.estimated_cost_usd:.5f})" if not self.is_empty else ""
        return (
            f"{self.input_tokens} in + {self.output_tokens} out"
            f" = {self.total_tokens} tokens{cost_label}{model_label}"
        )