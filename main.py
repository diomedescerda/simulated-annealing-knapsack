from models import KnapsackItem
from anneal import SimAnneal
import pandas as pd


def read_knapsack_data(path, max_capacity):
    df = pd.read_excel(path)
    items = [
        KnapsackItem(
            id=int(row['Id']),
            weight=float(row['Peso_kg']),
            value=int(row['Valor']),
            max_quantity=int(row['Cantidad'])
        )
        for _, row in df.iterrows()
    ]

    return items, max_capacity


if __name__ == "__main__":
    items, max_capacity = read_knapsack_data(
        'Mochila_capacidad_maxima_20kg.xlsx', 20)
    sa = SimAnneal(items=items, max_capacity=max_capacity)
    sa.anneal()
    print(f"Selected items: {[item.id for item, included in zip(items, sa.best_solution) if included]}")
    print(f"Solution: {sa.best_solution}")
    total_weight = sum(item.weight * count for item, count in zip(items, sa.best_solution))
    print(f"Total weight: {total_weight}/{sa.max_capacity}")
    sa.plot_learning()
