from dataclasses import dataclass

@dataclass
class KnapsackItem:
    id: int
    weight: float
    value: int
    max_quantity: int
