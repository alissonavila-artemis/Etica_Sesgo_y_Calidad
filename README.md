# Semana 4 – Ética, Sesgo, Calidad + XAI (Predicción de contratación)

## Objetivo
Construir un modelo de **clasificación supervisada** para predecir si un cliente contratará un producto financiero (`producto_contratado`), incorporando:
- calidad de datos (EDA),
- evaluación del modelo (CV + test),
- auditoría de sesgo por grupos,
- y técnicas XAI para explicar decisiones.


## Estructura del repositorio

---

## Metodología (resumen)
1. **EDA**: balance del target, distribuciones, correlación y nulos.  
2. **Preprocesamiento reproducible**: `ColumnTransformer + Pipeline` (imputación + escalado + one-hot).  
3. **Modelos**: Regresión Logística y Random Forest.  
4. **Evaluación**: validación cruzada 5-fold y métricas en test (Accuracy, Precision, Recall, F1, AUC).  
5. **Sesgo**: métricas por grupo (`genero`, `nivel_educacion`, `estado_civil`) y gaps (max-min).  
6. **XAI**: Permutation Importance + PDP + 1–2 ejemplos individuales.

---

## Principales hallazgos
- El dataset está **desbalanceado**: ~14% de casos positivos (`producto_contratado=1`).  
- En test, **Regresión Logística** mostró mejor **F1 y Recall** (más capacidad de detectar contrataciones) y **Random Forest** mejor **AUC** (separación global).  
- XAI confirma que el modelo se apoya principalmente en **ingresos_mensuales** y **uso digital** (app/banca online).  
- Se detectaron **diferencias por grupos** en métricas como Recall (TPR), por lo que se recomienda monitoreo de fairness y mitigación si se requiere.

