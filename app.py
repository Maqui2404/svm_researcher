import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time

st.title("Aplicación  SVM")
st.write("Marco Mayta - Facultad de Ingeniería Estadística e Informática - UNA PUNO")
st.write("Sube un archivo CSV, selecciona el separador, elige la columna objetivo, y configura el modelo SVM.")

separator = st.selectbox("Selecciona el separador de columnas en tu archivo CSV", [",", ";", " ", "|", "\t"], index=0)

uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=separator)
    st.write("Datos cargados:")
    st.write(df.head())
    st.write("Columnas disponibles:", df.columns.tolist())
    
    # Paso 3: Selección de columna objetivo y limpieza
    st.subheader("Paso 2: Selección de columna objetivo y limpieza de datos")
    target_column = st.selectbox("Selecciona la columna objetivo (clase a predecir)", df.columns)
    
    if target_column not in df.columns:
        st.error(f"La columna objetivo '{target_column}' no se encuentra en los datos. Por favor, verifica el archivo.")
    else:
        st.write("**Preprocesamiento de datos**")
        
        df_features = df.drop(columns=[target_column])
        
        if df_features.select_dtypes(include=['object']).empty:
            st.write("No se encontraron columnas categóricas, omitiendo One-Hot Encoding.")
        else:
            df_features = pd.get_dummies(df_features, drop_first=True)
            st.write("Datos después de aplicar One-Hot Encoding a las características:")
            st.write(df_features.head())
        
        df = pd.concat([df_features, df[target_column]], axis=1)
        
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Verificar que X no esté vacío después del procesamiento
        if X.shape[1] == 0:
            st.error("No hay columnas numéricas disponibles después de procesar los datos. Verifica el archivo y elige la columna objetivo correctamente.")
        else:
            # Normalizar las características
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Paso 5: Configuración avanzada del modelo SVM
            st.subheader("Paso 4: Configuración avanzada del modelo SVM")
            
            # Selección de kernel
            kernel_type = st.selectbox("Selecciona el tipo de kernel", ["linear", "rbf", "poly", "sigmoid"])
            
            # Parámetro de regularización C
            C_value = st.slider("Valor de penalización (C)", min_value=0.01, max_value=10.0, value=1.0)
            
            # Opciones específicas para RBF y Polinomial
            if kernel_type in ["rbf", "poly"]:
                gamma_value = st.slider("Valor de gamma", min_value=0.01, max_value=1.0, value=0.1, help="Controla la forma del kernel RBF o polinómico.")
            
            # Grado del polinomio (para el kernel polinomial)
            if kernel_type == "poly":
                degree_value = st.slider("Grado del polinomio", min_value=1, max_value=5, value=3)
            
            # Clase de peso balanceado para manejar clases desbalanceadas
            class_weight_opt = st.selectbox("¿Quieres balancear las clases?", [None, "balanced"], help="Usar 'balanced' ajusta el peso de las clases automáticamente.")

            # Paso 6: Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Paso 7: Entrenar el modelo SVM con indicador de progreso
            st.subheader("Paso 5: Entrenamiento y evaluación del modelo")
            
            # Configurar el modelo SVM según las opciones seleccionadas
            if kernel_type == "linear":
                model = SVC(kernel=kernel_type, C=C_value, class_weight=class_weight_opt)
            elif kernel_type == "rbf":
                model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value, class_weight=class_weight_opt)
            elif kernel_type == "poly":
                model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value, degree=degree_value, class_weight=class_weight_opt)
            else:  # "sigmoid"
                model = SVC(kernel=kernel_type, C=C_value, class_weight=class_weight_opt)
            
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
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            
            # Mostrar los resultados de evaluación
            st.write("**Resultados del modelo:**")
            st.dataframe(report_df)
            
            # Métricas específicas
            st.write(f"**Precisión (Accuracy)**: {report['accuracy']:.2f}")
            st.write(f"**Precisión (weighted)**: {report['weighted avg']['precision']:.2f}")
            st.write(f"**Recall (weighted)**: {report['weighted avg']['recall']:.2f}")
            st.write(f"**F1-Score (weighted)**: {report['weighted avg']['f1-score']:.2f}")
else:
    st.info("Por favor, sube un archivo CSV para continuar.")
