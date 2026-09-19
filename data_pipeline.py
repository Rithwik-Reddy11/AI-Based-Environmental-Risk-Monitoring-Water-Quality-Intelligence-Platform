from datetime import timedelta
import random
import numpy as np
import pandas as pd

BASE = {
    "chemical": {"pH": 5.5, "nitrate": 15, "temp": 22, "turbidity": 10, "do": 7, "conductivity": 300},
    "textile": {"pH": 7.5, "nitrate": 10, "temp": 28, "turbidity": 20, "do": 6.5, "conductivity": 500},
    "food_processing": {"pH": 6.8, "nitrate": 5, "temp": 18, "turbidity": 5, "do": 8, "conductivity": 150},
}


def generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    start = pd.Timestamp("2023-01-01")
    factories = [f"STATION-{i:02d}" for i in range(1, 11)]
    rows = []
    for i in range(n):
        t = start + timedelta(hours=i)
        industry = random.choice(list(BASE))
        b = BASE[industry]
        season = "summer" if 6 <= t.month <= 8 else "winter" if t.month in [12,1,2] else "transition"
        temp = b["temp"] + (5 if season == "summer" else -5 if season == "winter" else 0) + rng.normal(0, 1)
        ph = b["pH"] + rng.normal(0, .2)
        nitrate = b["nitrate"] + rng.normal(0, 1.5)
        turbidity = b["turbidity"] + rng.normal(0, 3)
        do = b["do"] + (-.5 if season == "summer" else .5 if season == "winter" else 0) + rng.normal(0, .5)
        conductivity = b["conductivity"] + rng.normal(0, 50)
        polluted = rng.random() < (.22 if industry == "chemical" else .17 if industry == "textile" else .10)
        if polluted:
            ph += rng.uniform(-1.8, 1.8); nitrate *= rng.uniform(1.5, 2.8); turbidity *= rng.uniform(1.3, 2.5)
            do += rng.uniform(-2.5, -.4); conductivity *= rng.uniform(1.3, 2.2)
        row = [t, random.choice(factories), industry, ph, turbidity, do, temp, conductivity, nitrate, int(polluted)]
        rows.append(row)
    df = pd.DataFrame(rows, columns=["Timestamp","Station","Industry_Type","pH","Turbidity","Dissolved_Oxygen","Water_Temperature","Conductivity","Nitrate","Pollution_Flag"])
    df[["pH","Turbidity","Dissolved_Oxygen","Water_Temperature","Conductivity","Nitrate"]] = df[["pH","Turbidity","Dissolved_Oxygen","Water_Temperature","Conductivity","Nitrate"]].clip(lower=0)
    return df


def load_data(path="data/synthetic_river_health_data.csv"):
    try:
        df = pd.read_csv(path)
        if "Timestamp" in df.columns: df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        elif df.index.name == "Timestamp": df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return generate_dataset()
