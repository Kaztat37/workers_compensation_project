"""Главная страница Streamlit-приложения.

Загрузка датасета Workers Compensation, предобработка, обучение четырёх
моделей регрессии (Linear, Ridge, Random Forest, XGBoost), вычисление
метрик MAE / MSE / RMSE / R², визуализация результатов и форма для
предсказания итоговой стоимости страхового возмещения.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


CATEGORICAL_COLUMNS: List[str] = [
    "Gender",
    "MaritalStatus",
    "PartTimeFullTime",
    "ClaimDescription",
]

NUMERICAL_FEATURES: List[str] = [
    "Age",
    "DependentChildren",
    "DependentsOther",
    "WeeklyPay",
    "HoursWorkedPerWeek",
    "DaysWorkedPerWeek",
    "InitialCaseEstimate",
    "AccidentMonth",
    "AccidentDayOfWeek",
    "ReportingDelay",
]

TARGET_COLUMN = "UltimateIncurredClaimCost"

LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "workers_compensation.csv")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Загружает датасет Workers Compensation.

    Сначала пытается прочитать локальный CSV ``data/workers_compensation.csv``
    (быстрее и работает оффлайн), при его отсутствии — скачивает датасет
    через ``fetch_openml`` (id=42876) и кеширует на диск.
    """

    if os.path.exists(LOCAL_CSV_PATH):
        return pd.read_csv(LOCAL_CSV_PATH)

    data = fetch_openml(data_id=42876, as_frame=True, parser="auto")
    df: pd.DataFrame = data.frame

    os.makedirs(os.path.dirname(LOCAL_CSV_PATH), exist_ok=True)
    try:
        df.to_csv(LOCAL_CSV_PATH, index=False)
    except OSError:
        # Если запись на диск недоступна, продолжаем работать с DataFrame в памяти.
        pass
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], StandardScaler]:
    """Предобработка: даты → признаки, кодирование категорий, масштабирование."""

    data = df.copy()

    data["DateTimeOfAccident"] = pd.to_datetime(data["DateTimeOfAccident"])
    data["DateReported"] = pd.to_datetime(data["DateReported"])

    data["AccidentMonth"] = data["DateTimeOfAccident"].dt.month
    data["AccidentDayOfWeek"] = data["DateTimeOfAccident"].dt.dayofweek
    data["ReportingDelay"] = (
        data["DateReported"] - data["DateTimeOfAccident"]
    ).dt.days

    data = data.drop(columns=["DateTimeOfAccident", "DateReported"])

    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        label_encoders[col] = encoder

    scaler = StandardScaler()
    data[NUMERICAL_FEATURES] = scaler.fit_transform(data[NUMERICAL_FEATURES])

    return data, label_encoders, scaler


@st.cache_resource(show_spinner=False)
def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log_target: bool = True,
) -> Dict[str, object]:
    """Обучает четыре модели регрессии и возвращает словарь моделей.

    Если ``log_target=True`` — каждая модель обучается на ``log1p(y)``;
    при предсказании в :func:`evaluate_models` применяется обратное
    преобразование ``expm1``. Это стандартная практика для целевых
    переменных с тяжёлым правым хвостом и значительно улучшает R².
    """

    target = np.log1p(y_train) if log_target else y_train

    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    for name, model in models.items():
        model.fit(X_train, target)
    return models


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_target: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Считает MAE/MSE/RMSE/R² и возвращает DataFrame метрик + предсказания.

    Если ``log_target=True`` — дополнительно считает R² на log-шкале,
    которая более устойчива для целевых переменных с длинным хвостом.
    """

    rows = []
    predictions: Dict[str, np.ndarray] = {}
    y_test_arr = y_test.to_numpy()
    log_y_test = np.log1p(y_test_arr) if log_target else None

    for name, model in models.items():
        raw_pred = model.predict(X_test)
        if log_target:
            # Клипуем лог-предсказания в разумном диапазоне, чтобы exp()
            # линейных моделей не «улетал» при выбросах.
            clipped = np.clip(raw_pred, a_min=0.0, a_max=float(np.log1p(y_test_arr.max() * 5)))
            y_pred = np.expm1(clipped)
        else:
            y_pred = raw_pred
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test_arr, y_pred)
        mse = mean_squared_error(y_test_arr, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test_arr, y_pred)
        row = {
            "Модель": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
        }
        if log_target and log_y_test is not None:
            row["R² (log)"] = r2_score(log_y_test, np.clip(raw_pred, a_min=0.0, a_max=None))
        rows.append(row)

    sort_key = "R² (log)" if log_target else "R²"
    metrics_df = (
        pd.DataFrame(rows).sort_values(sort_key, ascending=False).reset_index(drop=True)
    )
    return metrics_df, predictions


def plot_predictions_vs_actual(
    y_test: pd.Series, y_pred: np.ndarray, model_name: str, log_scale: bool = False
) -> plt.Figure:
    """Scatter-plot предсказаний модели против реальных значений."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=12)
    lo = max(float(min(y_test.min(), y_pred.min())), 1.0 if log_scale else 0.0)
    hi = float(max(y_test.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="y = x")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Реальные значения")
    ax.set_ylabel("Предсказанные значения")
    ax.set_title(f"{model_name}: предсказания vs реальные значения")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, top_n: int = 10
) -> plt.Figure:
    """Горизонтальный bar-chart важности признаков."""

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#3b82f6")
    ax.invert_yaxis()
    ax.set_xlabel("Важность")
    ax.set_title(f"Топ-{top_n} наиболее важных признаков")
    fig.tight_layout()
    return fig


def plot_target_distribution(y: pd.Series) -> plt.Figure:
    """Распределение целевой переменной (логарифмическая ось X)."""

    positive = y[y > 0]
    bins = np.logspace(
        np.log10(max(positive.min(), 1.0)),
        np.log10(positive.max()),
        50,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(positive, bins=bins, kde=False, ax=ax, color="#10b981")
    ax.set_xscale("log")
    ax.set_xlabel("UltimateIncurredClaimCost (лог. шкала)")
    ax.set_ylabel("Количество случаев")
    ax.set_title("Распределение итоговой стоимости страхового возмещения")
    fig.tight_layout()
    return fig


def build_input_dataframe(
    raw_input: Dict[str, object],
    label_encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Готовит одну строку с пользовательскими данными для предсказания."""

    row = {col: 0 for col in feature_columns}

    accident_dt = pd.to_datetime(raw_input["DateTimeOfAccident"])
    reported_dt = pd.to_datetime(raw_input["DateReported"])
    row["AccidentMonth"] = accident_dt.month
    row["AccidentDayOfWeek"] = accident_dt.dayofweek
    row["ReportingDelay"] = max((reported_dt - accident_dt).days, 0)

    row["Age"] = raw_input["Age"]
    row["DependentChildren"] = raw_input["DependentChildren"]
    row["DependentsOther"] = raw_input["DependentsOther"]
    row["WeeklyPay"] = raw_input["WeeklyPay"]
    row["HoursWorkedPerWeek"] = raw_input["HoursWorkedPerWeek"]
    row["DaysWorkedPerWeek"] = raw_input["DaysWorkedPerWeek"]
    row["InitialCaseEstimate"] = raw_input["InitialCaseEstimate"]

    for col in CATEGORICAL_COLUMNS:
        encoder = label_encoders[col]
        value = str(raw_input[col])
        if value in encoder.classes_:
            row[col] = int(encoder.transform([value])[0])
        else:
            # Неизвестная категория — используем самую частую (нулевой класс).
            row[col] = 0

    df_row = pd.DataFrame([row], columns=feature_columns)
    df_row[NUMERICAL_FEATURES] = scaler.transform(df_row[NUMERICAL_FEATURES])
    return df_row


def analysis_and_model_page() -> None:
    st.title("📊 Прогнозирование стоимости страховых выплат")
    st.caption(
        "Датасет Workers Compensation (OpenML id=42876, 100 000 записей). "
        "Целевая переменная — UltimateIncurredClaimCost."
    )

    with st.sidebar:
        st.header("Параметры обучения")
        sample_size = st.slider(
            "Размер выборки для обучения",
            min_value=5_000,
            max_value=100_000,
            value=30_000,
            step=5_000,
            help=(
                "Полный датасет — 100k записей. Меньшая выборка ускоряет "
                "обучение Random Forest и XGBoost; 30k — компромисс скорости "
                "и качества."
            ),
        )
        test_size = st.slider(
            "Доля тестовой выборки", min_value=0.1, max_value=0.4, value=0.2, step=0.05
        )
        log_target = st.checkbox(
            "Log-преобразование целевой переменной",
            value=True,
            help=(
                "Обучать модели на log1p(UltimateIncurredClaimCost). "
                "Резко повышает R² для распределения с длинным правым хвостом."
            ),
        )
        run_training = st.button("🚀 Загрузить данные и обучить модели", type="primary")

    if run_training:
        with st.spinner("Загружаем датасет (это может занять до минуты)…"):
            df = load_dataset()
        st.session_state["df"] = df
        st.session_state["sample_size"] = sample_size
        st.session_state["test_size"] = test_size
        st.session_state["log_target"] = log_target
        st.session_state["trained"] = False
        # Сбрасываем кеш ресурсов, чтобы модели переобучились с новыми параметрами.
        train_models.clear()

    if "df" not in st.session_state:
        st.info(
            "👈 Нажмите **«Загрузить данные и обучить модели»** в боковой панели, "
            "чтобы начать."
        )
        return

    df: pd.DataFrame = st.session_state["df"]

    st.subheader("1. Просмотр данных")
    c1, c2, c3 = st.columns(3)
    c1.metric("Записей", f"{len(df):,}".replace(",", " "))
    c2.metric("Признаков", df.shape[1] - 1)
    c3.metric(
        "Средняя сумма выплаты",
        f"${df[TARGET_COLUMN].mean():,.0f}".replace(",", " "),
    )

    with st.expander("Первые строки и описательная статистика", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.write(df.describe(include="all").transpose())

    st.subheader("2. Распределение целевой переменной")
    st.pyplot(plot_target_distribution(df[TARGET_COLUMN]))

    st.subheader("3. Предобработка")
    with st.spinner("Предобрабатываем данные…"):
        processed, label_encoders, scaler = preprocess_data(df)
    st.write("Пропущенные значения после предобработки:")
    missing = processed.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "Пропущенных значений нет ✅")

    sample_size = int(st.session_state.get("sample_size", 20_000))
    if sample_size < len(processed):
        processed_sample = processed.sample(n=sample_size, random_state=42)
    else:
        processed_sample = processed

    X = processed_sample.drop(columns=[TARGET_COLUMN])
    y = processed_sample[TARGET_COLUMN]
    feature_columns = X.columns.tolist()

    test_size = float(st.session_state.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write(
        f"Обучающая выборка: **{X_train.shape[0]:,}** записей · "
        f"Тестовая выборка: **{X_test.shape[0]:,}** записей".replace(",", " ")
    )

    st.subheader("4. Обучение моделей")
    use_log = bool(st.session_state.get("log_target", True))
    with st.spinner("Обучаем Linear / Ridge / Random Forest / XGBoost…"):
        models = train_models(X_train, y_train, log_target=use_log)
    st.success(
        "Все модели обучены"
        + (" (целевая переменная — log1p)." if use_log else " на исходной шкале.")
    )

    st.subheader("5. Сравнение моделей")
    metrics_df, predictions = evaluate_models(
        models, X_test, y_test, log_target=use_log
    )

    fmt = {
        "MAE": "${:,.0f}",
        "MSE": "{:,.0f}",
        "RMSE": "${:,.0f}",
        "R²": "{:.4f}",
        "R² (log)": "{:.4f}",
    }
    available_fmt = {k: v for k, v in fmt.items() if k in metrics_df.columns}
    gradient_col = "R² (log)" if "R² (log)" in metrics_df.columns else "R²"
    st.dataframe(
        metrics_df.style.format(available_fmt).background_gradient(
            subset=[gradient_col], cmap="Greens"
        ),
        use_container_width=True,
    )

    best_model_name = metrics_df.iloc[0]["Модель"]
    st.success(f"🏆 Лучшая модель по {gradient_col}: **{best_model_name}**")

    st.subheader("6. Предсказания vs реальные значения")
    selected_model_name = st.selectbox(
        "Выберите модель для визуализации",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )
    st.pyplot(
        plot_predictions_vs_actual(
            y_test,
            predictions[selected_model_name],
            selected_model_name,
            log_scale=use_log,
        )
    )

    st.subheader("7. Важность признаков")
    rf_model: RandomForestRegressor = models["Random Forest"]  # type: ignore[assignment]
    st.pyplot(
        plot_feature_importance(feature_columns, rf_model.feature_importances_)
    )

    # Сохраняем артефакты в session_state — пригодится для формы предсказания.
    st.session_state["models"] = models
    st.session_state["label_encoders"] = label_encoders
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = feature_columns
    st.session_state["best_model_name"] = best_model_name
    st.session_state["raw_df"] = df
    st.session_state["trained"] = True

    st.subheader("8. Предсказание для нового случая")
    if not st.session_state.get("trained"):
        st.info("Сначала обучите модели.")
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    encoders: Dict[str, LabelEncoder] = st.session_state["label_encoders"]
    fitted_scaler: StandardScaler = st.session_state["scaler"]
    feature_cols: List[str] = st.session_state["feature_columns"]

    with st.form("prediction_form"):
        st.markdown("Заполните параметры случая и получите прогноз итоговой выплаты.")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Возраст работника", min_value=13, max_value=80, value=35)
            gender = st.selectbox(
                "Пол",
                options=sorted(encoders["Gender"].classes_.tolist()),
            )
            marital = st.selectbox(
                "Семейное положение",
                options=sorted(encoders["MaritalStatus"].classes_.tolist()),
            )
            part_time = st.selectbox(
                "Тип занятости",
                options=sorted(encoders["PartTimeFullTime"].classes_.tolist()),
            )
            children = st.number_input(
                "Детей на иждивении", min_value=0, max_value=10, value=0
            )
            other_dep = st.number_input(
                "Других иждивенцев", min_value=0, max_value=10, value=0
            )
        with col2:
            weekly_pay = st.number_input(
                "Еженедельная зарплата ($)", min_value=0.0, value=500.0, step=10.0
            )
            hours = st.number_input(
                "Часов работы в неделю", min_value=0, max_value=120, value=38
            )
            days = st.number_input(
                "Дней работы в неделю", min_value=1, max_value=7, value=5
            )
            initial_estimate = st.number_input(
                "Начальная оценка случая ($)",
                min_value=0.0,
                value=5_000.0,
                step=100.0,
            )

            default_classes = encoders["ClaimDescription"].classes_.tolist()
            common_descriptions = sorted(default_classes)[:50]
            description = st.selectbox(
                "Описание заявки (ClaimDescription)", options=common_descriptions
            )
            accident_date = st.date_input(
                "Дата несчастного случая",
                value=pd.Timestamp("2000-01-15").date(),
            )
            reported_date = st.date_input(
                "Дата сообщения о случае",
                value=pd.Timestamp("2000-01-20").date(),
            )

        chosen_model = st.selectbox(
            "Модель для предсказания",
            options=list(st.session_state["models"].keys()),
            index=list(st.session_state["models"].keys()).index(
                st.session_state["best_model_name"]
            ),
        )

        submit = st.form_submit_button("🔮 Предсказать стоимость", type="primary")

    if submit:
        raw_input = {
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital,
            "PartTimeFullTime": part_time,
            "DependentChildren": children,
            "DependentsOther": other_dep,
            "WeeklyPay": weekly_pay,
            "HoursWorkedPerWeek": hours,
            "DaysWorkedPerWeek": days,
            "InitialCaseEstimate": initial_estimate,
            "ClaimDescription": description,
            "DateTimeOfAccident": accident_date,
            "DateReported": reported_date,
        }
        input_df = build_input_dataframe(raw_input, encoders, fitted_scaler, feature_cols)
        model = st.session_state["models"][chosen_model]
        raw_prediction = float(model.predict(input_df)[0])
        if st.session_state.get("log_target", True):
            prediction = float(np.expm1(raw_prediction))
        else:
            prediction = raw_prediction
        prediction = max(prediction, 0.0)
        st.metric(
            f"Прогноз UltimateIncurredClaimCost ({chosen_model})",
            f"${prediction:,.2f}".replace(",", " "),
        )
        st.caption(
            "Внимание: модель обучена на исторических данных и предоставляет "
            "оценочный прогноз. Для бизнес-решений требуется дополнительная валидация."
        )


# Точка входа для st.Page — Streamlit вызывает модуль как скрипт.
analysis_and_model_page()
