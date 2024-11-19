
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
import pickle
import base64
import io

st.title("Aplicación  SVM")
st.write("Marco Mayta - Facultad de Ingeniería Estadística e Informática - UNA PUNO")
st.write("Sube un archivo CSV, selecciona el separador, elige la columna objetivo, y configura el modelo SVM.")

tab1, tab2 = st.tabs(["Entrenar Nuevo Modelo", "Cargar Modelo Existente"])

with tab1:
    separator = st.selectbox("Selecciona el separador de columnas en tu archivo CSV", [
                             ",", ";", " ", "|", "\t"], index=0)

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV", type="csv", key="train_data")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=separator)
        st.write("Datos cargados:")
        st.write(df.head())
        st.write("Columnas disponibles:", df.columns.tolist())

        st.subheader(
            "Paso 2: Selección de columna objetivo y limpieza de datos")
        target_column = st.selectbox(
            "Selecciona la columna objetivo (clase a predecir)", df.columns)

        if target_column not in df.columns:
            st.error(
                f"La columna objetivo '{target_column}' no se encuentra en los datos. Por favor, verifica el archivo.")
        else:
            st.write("**Preprocesamiento de datos**")

            df_features = df.drop(columns=[target_column])

            if df_features.select_dtypes(include=['object']).empty:
                st.write(
                    "No se encontraron columnas categóricas, omitiendo One-Hot Encoding.")
            else:
                df_features = pd.get_dummies(df_features, drop_first=True)
                st.write(
                    "Datos después de aplicar One-Hot Encoding a las características:")
                st.write(df_features.head())

            df = pd.concat([df_features, df[target_column]], axis=1)

            X = df.drop(columns=[target_column]).values
            y = df[target_column].values

            if X.shape[1] == 0:
                st.error(
                    "No hay columnas numéricas disponibles después de procesar los datos. Verifica el archivo y elige la columna objetivo correctamente.")
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                st.subheader("Paso 4: Configuración avanzada del modelo SVM")

                kernel_type = st.selectbox("Selecciona el tipo de kernel", [
                                           "linear", "rbf", "poly", "sigmoid"])

                C_value = st.slider("Valor de penalización (C)",
                                    min_value=0.01, max_value=10.0, value=1.0)

                if kernel_type in ["rbf", "poly"]:
                    gamma_value = st.slider("Valor de gamma", min_value=0.01, max_value=1.0,
                                            value=0.1, help="Controla la forma del kernel RBF o polinómico.")

                if kernel_type == "poly":
                    degree_value = st.slider(
                        "Grado del polinomio", min_value=1, max_value=5, value=3)

                class_weight_opt = st.selectbox("¿Quieres balancear las clases?", [
                                                None, "balanced"], help="Usar 'balanced' ajusta el peso de las clases automáticamente.")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42)

                st.subheader("Paso 5: Entrenamiento y evaluación del modelo")

                if kernel_type == "linear":
                    model = SVC(kernel=kernel_type, C=C_value,
                                class_weight=class_weight_opt)
                elif kernel_type == "rbf":
                    model = SVC(kernel=kernel_type, C=C_value,
                                gamma=gamma_value, class_weight=class_weight_opt)
                elif kernel_type == "poly":
                    model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value,
                                degree=degree_value, class_weight=class_weight_opt)
                else:  
                    model = SVC(kernel=kernel_type, C=C_value,
                                class_weight=class_weight_opt)

                with st.spinner('Entrenando modelo...'):
                    progress = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress.progress(percent_complete + 1)

                    model.fit(X_train, y_train)

                st.success("Entrenamiento completado")

                y_pred = model.predict(X_test)
                report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()

                st.write("**Resultados del modelo:**")
                st.dataframe(report_df)

                st.write(f"**Precisión (Accuracy)**: {report['accuracy']:.2f}")
                st.write(
                    f"**Precisión (weighted)**: {report['weighted avg']['precision']:.2f}")
                st.write(
                    f"**Recall (weighted)**: {report['weighted avg']['recall']:.2f}")
                st.write(
                    f"**F1-Score (weighted)**: {report['weighted avg']['f1-score']:.2f}")

                st.subheader("Descargar Modelo Entrenado")

                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': df_features.columns.tolist(),
                    'target_column': target_column,
                    'original_columns': list(df.drop(columns=[target_column]).columns),
                    'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                    'numeric_columns': list(df.select_dtypes(exclude=['object']).columns)
                }

                buff = io.BytesIO()
                pickle.dump(model_data, buff)
                buff.seek(0)

                b64 = base64.b64encode(buff.read()).decode()
                href = f'<a href="data:file/pickle;base64,{b64}" download="modelo_svm.pkl">Descargar Modelo Entrenado</a>'
                st.markdown(href, unsafe_allow_html=True)

                st.subheader("Probar Modelo - Entrada Manual")

                with st.form(key='prediction_form'):
                    st.write("Ingrese los valores para realizar una predicción:")

                    input_data = {}
                    for col in df.drop(columns=[target_column]).columns:
                        if col in df.select_dtypes(include=['object']).columns:
                            unique_values = df[col].unique().tolist()
                            input_data[col] = st.selectbox(
                                f"Seleccione valor para {col}", unique_values)
                        else:
                            input_data[col] = st.number_input(
                                f"Ingrese valor para {col}", value=0.0)

                    submit_button = st.form_submit_button(
                        label='Realizar Predicción')

                if submit_button:
                    input_df = pd.DataFrame([input_data])

                    if not input_df.select_dtypes(include=['object']).empty:
                        input_df = pd.get_dummies(input_df, drop_first=True)

                    for col in df_features.columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[df_features.columns]

                    X_new = scaler.transform(input_df.values)

                    prediction = model.predict(X_new)

                    st.success(f"La predicción es: {prediction[0]}")
    else:
        st.info("Por favor, sube un archivo CSV para continuar.")


with tab2:
    st.subheader("Cargar Modelo Existente")

    uploaded_model = st.file_uploader(
        "Sube un modelo guardado (.pkl)", type="pkl")

    if uploaded_model is not None:
        try:
            model_data = pickle.load(uploaded_model)
            loaded_model = model_data['model']
            loaded_scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']

            if 'original_columns' in model_data:
                original_columns = model_data['original_columns']
                categorical_columns = model_data.get('categorical_columns', [])
                numeric_columns = model_data.get('numeric_columns', [])
            else:
                original_columns = feature_names
                categorical_columns = []
                numeric_columns = original_columns

            st.success("Modelo cargado correctamente")

            st.subheader("Probar Modelo Cargado - Entrada Manual")

            with st.form(key='loaded_model_form'):
                st.write("Ingrese los valores para realizar una predicción:")

                input_data = {}
                for col in original_columns:
                    if col in categorical_columns:
                        input_data[col] = st.text_input(
                            f"Ingrese valor para {col}")
                    else:
                        input_data[col] = st.number_input(
                            f"Ingrese valor para {col}", value=0.0)

                submit_button = st.form_submit_button(
                    label='Realizar Predicción')

            if submit_button:
                input_df = pd.DataFrame([input_data])

                if categorical_columns:
                    input_df = pd.get_dummies(input_df, drop_first=True)

                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_names]

                X_new = loaded_scaler.transform(input_df.values)

                prediction = loaded_model.predict(X_new)

                st.success(f"La predicción es: {prediction[0]}")

        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            st.write(
                "Por favor, asegúrese de que el archivo .pkl sea un modelo SVM válido.")
