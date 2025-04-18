from anneal import SimAnneal, KnapsackItem
import pandas as pd

# Create test items
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


items = [
    KnapsackItem(id=1, weight=3, value=60, max_quantity=20),  # A
    KnapsackItem(id=2, weight=4, value=70, max_quantity=17.5),  # A
    KnapsackItem(id=3, weight=5, value=90, max_quantity=18)  # B
]

items_dos = [
    KnapsackItem(id=1, weight=6, value=100, max_quantity=16.66),  # A
    KnapsackItem(id=2, weight=5, value=90, max_quantity=18),  # A
    KnapsackItem(id=3, weight=4, value=80, max_quantity=20)  # B
]


# Initialize with capacity = 10
sa = SimAnneal(items=items, max_capacity=10)

for i in range(1):
    solution, fitness = sa.initial_solution()
    total_weight = sum(item.weight * count for item, count in zip(items, solution))
    print(f"Test {i + 1}:")
    print(f"Solution: {solution}")
    print(f"Selected items: {[item.id for item, included in zip(items, solution) if included]}")
    print(f"Total value: {fitness}")
    print(f"Total weight: {total_weight}/{sa.max_capacity}")
    print(f"Valid?: {total_weight <= sa.max_capacity}\n")

    best = SimAnneal(items=items, max_capacity=10)
    best.anneal()
    print(f"\nSolution: {best.best_solution}")
    print(f"Total value: {best.best_fitness}")
