import pandas as pd

def create_pso_results_table(iterations, particles, w, c1, c2, mape):
    data = {
        "Parameter": ["Iterations", "Particles", "Inertia Weight (w)", "C1", "C2", "MAPE"],
        "Value": [iterations, particles, w, c1, c2, mape]
    }
    df = pd.DataFrame(data)
    return df
