import pandas as pd

def preprocess_data(df):
    required_columns = ['Tahun', 'Bulan', 'Nilai Impor migas (Juta US$)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_columns)}")
    
    # Memastikan kolom Tahun dan Bulan mengandung nilai numerik
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
    df['Bulan'] = pd.to_numeric(df['Bulan'], errors='coerce')

    # Menampilkan baris dengan nilai non-numerik atau null
    invalid_rows = df[df['Tahun'].isnull() | df['Bulan'].isnull()]
    if not invalid_rows.empty:
        raise ValueError(f"Kolom 'Tahun' atau 'Bulan' mengandung nilai non-numerik atau null. Baris yang bermasalah:\n{invalid_rows}")

    if not ((df['Bulan'] >= 1) & (df['Bulan'] <= 12)).all():
        raise ValueError("Kolom 'Bulan' harus berisi nilai antara 1 dan 12.")

    try:
        df['Tanggal'] = pd.to_datetime(df[['Tahun', 'Bulan']].assign(DAY=1))
    except Exception as e:
        raise ValueError(f"Terdapat masalah dalam mengubah kolom 'Tahun' dan 'Bulan' menjadi format datetime: {e}")

    df = df.set_index('Tanggal')
    df = df[['Nilai Impor migas (Juta US$)']]
    return df

def fuzzy_time_series(data, order):
    fuzzy_series = []
    for i in range(len(data) - order):
        pattern = tuple(data[i:i + order])
        next_value = data[i + order]
        fuzzy_series.append((pattern, next_value))
    return fuzzy_series

def build_transition_matrix(fuzzy_series, order):
    from collections import defaultdict
    transitions = defaultdict(list)
    for pattern, next_value in fuzzy_series:
        transitions[pattern].append(next_value)

    transition_matrix = {}
    for pattern, next_values in transitions.items():
        transition_matrix[pattern] = sum(next_values) / len(next_values)
    
    return transition_matrix

def forecast(data, transition_matrix, order, steps):
    forecasts = []
    for i in range(steps):
        current_pattern = tuple(data[-order:])
        next_value = transition_matrix.get(current_pattern, data[-1])
        forecasts.append(next_value)
        data = data[1:] + [next_value]
    return forecasts
