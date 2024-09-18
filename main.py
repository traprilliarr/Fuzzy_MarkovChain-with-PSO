import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from markov_chain import preprocess_data, fuzzy_time_series, build_transition_matrix, forecast
from pso import create_pso_results_table

# Menggunakan Streamlit untuk GUI
st.set_page_config(page_title='Peramalan nilai impor migas di Indonesia menggunakan metode Fuzzy Time Series Model Markov Chain dengan algoritma Particle Swarm Optimization untuk perdagangan internasional')

st.markdown(
    """
    <style>
    .st-ba {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .st-cj {
        font-size: 24px;
        font-weight: bold;
        color: #336699;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Emoticon untuk judul
st.title("📊 Peramalan Nilai Impor Migas di Indonesia Menggunakan Metode Fuzzy Time Series Model Markov Chain Dengan Algoritma Particle Swarm Optimization Untuk Perdagangan Internasional")

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Load Data", "Hasil Peramalan"])

if page == "Load Data":
    st.title("Load Data")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Kolom dalam data:", df.columns)
            st.write("5 baris pertama data:", df.head())

            df = preprocess_data(df)
            st.session_state['data'] = df
            st.write("Data yang telah diproses:")
            st.dataframe(df)
            st.success("File berhasil diupload dan diproses")
        except ValueError as e:
            st.error(f"Error: {e}")

if page == "Hasil Peramalan":
    st.title("Hasil Peramalan Nilai Impor Migas")
    if 'data' in st.session_state:
        df = st.session_state['data']

        st.header("Pengaturan Parameter")
        with st.form(key='parameters_form'):
            order = st.slider("Order", 1, 10, 3)
            iterations = st.slider("Jumlah Iterasi", 1, 100, 10)
            particles = st.slider("Jumlah Partikel", 1, 100, 30)
            w = st.slider("Bobot Inersia (w)", 0.0, 1.0, 0.5)
            c1 = st.slider("Nilai C1", 0.0, 2.0, 1.5)
            c2 = st.slider("Nilai C2", 0.0, 2.0, 1.5)
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]['Nilai Impor migas (Juta US$)']
            test_data = df.iloc[train_size:]['Nilai Impor migas (Juta US$)']
            
            fuzzy_series = fuzzy_time_series(train_data.values, order)
            transition_matrix = build_transition_matrix(fuzzy_series, order)
            steps = len(test_data)
            forecasts = forecast(train_data.values, transition_matrix, order, steps)

            st.session_state['forecasts'] = forecasts
            st.session_state['train_data'] = train_data
            st.session_state['test_data'] = test_data
            st.session_state['fuzzy_series'] = fuzzy_series
            st.session_state['transition_matrix'] = transition_matrix
            st.session_state['order'] = order
            st.session_state['iterations'] = iterations
            st.session_state['particles'] = particles
            st.session_state['w'] = w
            st.session_state['c1'] = c1
            st.session_state['c2'] = c2

    if 'forecasts' in st.session_state:
        forecasts = st.session_state['forecasts']
        train_data = st.session_state['train_data']
        test_data = st.session_state['test_data']
        fuzzy_series = st.session_state['fuzzy_series']
        transition_matrix = st.session_state['transition_matrix']
        order = st.session_state['order']
        iterations = st.session_state['iterations']
        particles = st.session_state['particles']
        w = st.session_state['w']
        c1 = st.session_state['c1']
        c2 = st.session_state['c2']

        st.subheader("Tabel III-2: Tabel Fuzzy Logic Relationship (FLR)")
        with st.expander("Lihat Tabel FLR"):
            flr_table = pd.DataFrame({'Periode': range(1, len(fuzzy_series) + 1),
                                      'Nilai Impor (US dollar)': [pattern for pattern, _ in fuzzy_series],
                                      'FLR': [next_value for _, next_value in fuzzy_series]})
            st.dataframe(flr_table)
            st.session_state['flr_table'] = flr_table

        st.subheader("Tabel III-3: Pemetaan Nilai Linguistik")
        with st.expander("Lihat Tabel Pemetaan"):
            all_values = np.concatenate((train_data.values, test_data.values))
            forecasts_full = np.concatenate((forecasts, [np.nan] * (len(all_values) - len(forecasts))))
            linguistic_table = pd.DataFrame({
                'Periode': range(1, len(all_values) + 1),
                'Nilai Impor (US dollar)': all_values,
                'Fuzzifikasi': ['A' if i < len(train_data) else 'B' for i in range(len(all_values))],
                'Hasil': forecasts_full
            })
            st.dataframe(linguistic_table)
            st.session_state['linguistic_table'] = linguistic_table

        st.subheader("Tabel III-4: Hasil Pengujian Data")
        with st.expander("Lihat Tabel Hasil"):
            error_table = pd.DataFrame({
                'Periode': range(1, len(test_data) + 1),
                'Data Aktual': test_data.values,
                'Peramalan': forecasts,
                'Error': [actual - forecast for actual, forecast in zip(test_data.values, forecasts)]
            })
            st.dataframe(error_table)
            st.session_state['error_table'] = error_table

        st.subheader("Hasil Peramalan")
        st.line_chart({
            'Nilai Impor Aktual': np.concatenate((train_data.values, test_data.values)),
            'Peramalan': np.concatenate((train_data.values, forecasts))
        })

        fig, ax = plt.subplots()
        ax.plot(train_data.index, train_data, label='Data Latih')
        ax.plot(test_data.index, test_data, label='Data Uji')
        forecast_index = pd.date_range(start=test_data.index[0], periods=len(forecasts), freq='MS')
        ax.plot(forecast_index, forecasts, label='Peramalan', color='red', linestyle='--')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Nilai Impor migas (Juta US$)')
        ax.legend()
        st.pyplot(fig)
