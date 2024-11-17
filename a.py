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

# Crear pestañas para separar la funcionalidad
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

        # Paso 3: Selección de columna objetivo y limpieza
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

            # Verificar que X no esté vacío después del procesamiento
            if X.shape[1] == 0:
                st.error(
                    "No hay columnas numéricas disponibles después de procesar los datos. Verifica el archivo y elige la columna objetivo correctamente.")
            else:
                # Normalizar las características
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Paso 5: Configuración avanzada del modelo SVM
                st.subheader("Paso 4: Configuración avanzada del modelo SVM")

                # Selección de kernel
                kernel_type = st.selectbox("Selecciona el tipo de kernel", [
                                           "linear", "rbf", "poly", "sigmoid"])

                # Parámetro de regularización C
                C_value = st.slider("Valor de penalización (C)",
                                    min_value=0.01, max_value=10.0, value=1.0)

                # Opciones específicas para RBF y Polinomial
                if kernel_type in ["rbf", "poly"]:
                    gamma_value = st.slider("Valor de gamma", min_value=0.01, max_value=1.0,
                                            value=0.1, help="Controla la forma del kernel RBF o polinómico.")

                # Grado del polinomio (para el kernel polinomial)
                if kernel_type == "poly":
                    degree_value = st.slider(
                        "Grado del polinomio", min_value=1, max_value=5, value=3)

                # Clase de peso balanceado para manejar clases desbalanceadas
                class_weight_opt = st.selectbox("¿Quieres balancear las clases?", [
                                                None, "balanced"], help="Usar 'balanced' ajusta el peso de las clases automáticamente.")

                # Paso 6: Dividir los datos en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42)

                # Paso 7: Entrenar el modelo SVM con indicador de progreso
                st.subheader("Paso 5: Entrenamiento y evaluación del modelo")

                # Configurar el modelo SVM según las opciones seleccionadas
                if kernel_type == "linear":
                    model = SVC(kernel=kernel_type, C=C_value,
                                class_weight=class_weight_opt)
                elif kernel_type == "rbf":
                    model = SVC(kernel=kernel_type, C=C_value,
                                gamma=gamma_value, class_weight=class_weight_opt)
                elif kernel_type == "poly":
                    model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value,
                                degree=degree_value, class_weight=class_weight_opt)
                else:  # "sigmoid"
                    model = SVC(kernel=kernel_type, C=C_value,
                                class_weight=class_weight_opt)

                with st.spinner('Entrenando modelo...'):
                    progress = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress.progress(percent_complete + 1)

                    # Entrenar el modelo
                    model.fit(X_train, y_train)

                st.success("Entrenamiento completado")

                # Evaluar el modelo
                y_pred = model.predict(X_test)
                report = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()

                # Mostrar los resultados de evaluación
                st.write("**Resultados del modelo:**")
                st.dataframe(report_df)

                # Métricas específicas
                st.write(f"**Precisión (Accuracy)**: {report['accuracy']:.2f}")
                st.write(
                    f"**Precisión (weighted)**: {report['weighted avg']['precision']:.2f}")
                st.write(
                    f"**Recall (weighted)**: {report['weighted avg']['recall']:.2f}")
                st.write(
                    f"**F1-Score (weighted)**: {report['weighted avg']['f1-score']:.2f}")

                # Añadir sección para descargar el modelo
                st.subheader("Descargar Modelo Entrenado")

                # Guardar el modelo y el scaler
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': df_features.columns.tolist(),
                    'target_column': target_column,
                    # Guardar columnas originales
                    'original_columns': list(df.drop(columns=[target_column]).columns),
                    # Guardar columnas categóricas
                    'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                    # Guardar columnas numéricas
                    'numeric_columns': list(df.select_dtypes(exclude=['object']).columns)
                }

                # Crear botón de descarga
                buff = io.BytesIO()
                pickle.dump(model_data, buff)
                buff.seek(0)

                # Crear link de descarga
                b64 = base64.b64encode(buff.read()).decode()
                href = f'<a href="data:file/pickle;base64,{b64}" download="modelo_svm.pkl">Descargar Modelo Entrenado</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Sección para probar el modelo con datos manuales
                st.subheader("Probar Modelo - Entrada Manual")

                # Crear formulario para entrada de datos
                with st.form(key='prediction_form'):
                    st.write("Ingrese los valores para realizar una predicción:")

                    # Crear campos de entrada para cada característica
                    input_data = {}
                    for col in df.drop(columns=[target_column]).columns:
                        if col in df.select_dtypes(include=['object']).columns:
                            # Para columnas categóricas, crear un selectbox con valores únicos
                            unique_values = df[col].unique().tolist()
                            input_data[col] = st.selectbox(
                                f"Seleccione valor para {col}", unique_values)
                        else:
                            # Para columnas numéricas, crear un número input
                            input_data[col] = st.number_input(
                                f"Ingrese valor para {col}", value=0.0)

                    submit_button = st.form_submit_button(
                        label='Realizar Predicción')

                if submit_button:
                    # Crear DataFrame con los datos ingresados
                    input_df = pd.DataFrame([input_data])

                    # Aplicar el mismo preprocesamiento
                    if not input_df.select_dtypes(include=['object']).empty:
                        input_df = pd.get_dummies(input_df, drop_first=True)

                    # Asegurar que tengamos las mismas columnas que en el entrenamiento
                    for col in df_features.columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[df_features.columns]

                    # Escalar los datos
                    X_new = scaler.transform(input_df.values)

                    # Realizar predicción
                    prediction = model.predict(X_new)

                    # Mostrar resultado
                    st.success(f"La predicción es: {prediction[0]}")
    else:
        st.info("Por favor, sube un archivo CSV para continuar.")

# [Todo el código anterior se mantiene igual hasta la parte de cargar el modelo]

with tab2:
    st.subheader("Cargar Modelo Existente")

    # Cargar modelo existente
    uploaded_model = st.file_uploader(
        "Sube un modelo guardado (.pkl)", type="pkl")

    if uploaded_model is not None:
        try:
            # Cargar el modelo
            model_data = pickle.load(uploaded_model)
            loaded_model = model_data['model']
            loaded_scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']

            # Manejo de compatibilidad con modelos antiguos
            if 'original_columns' in model_data:
                original_columns = model_data['original_columns']
                categorical_columns = model_data.get('categorical_columns', [])
                numeric_columns = model_data.get('numeric_columns', [])
            else:
                # Si es un modelo antiguo, usar feature_names como columnas originales
                original_columns = feature_names
                categorical_columns = []
                numeric_columns = original_columns  # Asumir que todas son numéricas

            st.success("Modelo cargado correctamente")

            # Sección para probar el modelo cargado con datos manuales
            st.subheader("Probar Modelo Cargado - Entrada Manual")

            # Crear formulario para entrada de datos
            with st.form(key='loaded_model_form'):
                st.write("Ingrese los valores para realizar una predicción:")

                # Crear campos de entrada para cada característica
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
                # Crear DataFrame con los datos ingresados
                input_df = pd.DataFrame([input_data])

                # Aplicar el mismo preprocesamiento solo si hay columnas categóricas
                if categorical_columns:
                    input_df = pd.get_dummies(input_df, drop_first=True)

                # Asegurar que tengamos las mismas columnas que en el entrenamiento
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_names]

                # Escalar los datos
                X_new = loaded_scaler.transform(input_df.values)

                # Realizar predicción
                prediction = loaded_model.predict(X_new)

                # Mostrar resultado
                st.success(f"La predicción es: {prediction[0]}")

        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            st.write(
                "Por favor, asegúrese de que el archivo .pkl sea un modelo SVM válido.")
