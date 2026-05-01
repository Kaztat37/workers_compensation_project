"""Страница презентации проекта (streamlit-reveal-slides).

Все переключатели в сайдбаре действительно переинициализируют слайды:
- Тема — выбирается из списка CSS, которые поставляются вместе с пакетом.
- Высота слайдов — пересчитывается при ресайзе iframe.
- Переход — передаётся в config Reveal.js.
- Плагины — отображаемые имена мапятся на реальные ключи Reveal
  (`RevealHighlight`, `RevealSearch`, `RevealNotes`, `RevealZoom`,
  `RevealMath.KaTeX`).

Чтобы изменения переходов и плагинов гарантированно применялись, мы
передаём в `rs.slides(...)` параметр `key`, зависящий от текущих настроек:
при его изменении Streamlit ремонтирует компонент, и Reveal.js
инициализируется заново с новыми параметрами.
"""

from __future__ import annotations

import streamlit as st

try:
    import reveal_slides as rs

    REVEAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    REVEAL_AVAILABLE = False


# Темы, которые реально поставляются вместе с пакетом streamlit-reveal-slides
# (см. frontend/build/static/css). Тема меняет фон, шрифты и цвета слайдов.
AVAILABLE_THEMES = [
    "black",
    "white",
    "league",
    "beige",
    "sky",
    "night",
    "moon",
    "solarized",
    "serif",
    "simple",
    "blood",
    "dracula",
    "black-contrast",
    "white-contrast",
]

# Виды переходов Reveal.js (передаются в config).
AVAILABLE_TRANSITIONS = ["none", "fade", "slide", "convex", "concave", "zoom"]

# Плагины Reveal.js, доступные в встроенной сборке streamlit-reveal-slides.
# Ключ — то, что видит пользователь; значение — имя плагина для Reveal.
PLUGIN_OPTIONS: dict[str, str] = {
    "Подсветка кода (highlight)": "RevealHighlight",
    "Поиск (search)": "RevealSearch",
    "Заметки докладчика (notes)": "RevealNotes",
    "Зум по двойному клику (zoom)": "RevealZoom",
    "Формулы KaTeX (math)": "RevealMath.KaTeX",
}


PRESENTATION_MARKDOWN = r"""
# Прогнозирование стоимости страховых выплат
### Workers Compensation · UltimateIncurredClaimCost
---
## Введение
- Анализ данных о страховых случаях компенсации работникам.
- Цель — предсказать **итоговую** стоимость страхового возмещения.
- Датасет: Workers Compensation (OpenML id = 42876, **100 000** записей).
---
## Бизнес-задача
- Страховые компании нуждаются в точной оценке будущих выплат.
- Начальная оценка (`InitialCaseEstimate`) часто отличается от итоговой.
- Точные прогнозы помогают формировать резервы и тарифицировать риски.
---
## Описание данных
- 13 признаков + целевая переменная `UltimateIncurredClaimCost`.
- Категориальные: `Gender`, `MaritalStatus`, `PartTimeFullTime`, `ClaimDescription`.
- Datetime: `DateTimeOfAccident`, `DateReported`.
- Числовые: возраст, зарплата, часы, начальная оценка и др.
---
## Этапы работы
1. Загрузка данных через `sklearn.datasets.fetch_openml`.
2. Предобработка: даты → `AccidentMonth`, `AccidentDayOfWeek`, `ReportingDelay`.
3. `LabelEncoder` для категорий, `StandardScaler` для числовых.
4. `train_test_split` 80/20.
5. Обучение Linear / Ridge / Random Forest / XGBoost.
6. Оценка MAE, MSE, RMSE, R² и анализ важности признаков.
---
## Пример формулы для KaTeX
Среднеквадратичная ошибка:
$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
> 💡 Чтобы формула отрисовалась, включите плагин «Формулы KaTeX (math)».
---
## Пример блока кода (highlight)
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```
> 💡 Включите плагин «Подсветка кода (highlight)», чтобы код подсветился.
---
## Ключевые признаки
- **InitialCaseEstimate** — обычно №1 по важности.
- **WeeklyPay**, **Age**, **HoursWorkedPerWeek**.
- **ReportingDelay** — задержка отчёта о случае.
- **ClaimDescription** — тип травмы.
---
## Сравнение моделей
| Модель              | Сильные стороны                              |
|---------------------|----------------------------------------------|
| Linear Regression   | Простота, интерпретируемость                 |
| Ridge Regression    | Регуляризация, мультиколлинеарность          |
| Random Forest       | Устойчивость, нелинейности, важность фич     |
| XGBoost             | Высокая точность, регуляризация              |
---
## Streamlit-приложение
- Многостраничное приложение на `st.navigation` + `st.Page`.
- Загрузка данных, обучение и метрики «в один клик».
- Графики: распределение цели, предсказания vs факт, важность признаков.
- Форма для интерактивного предсказания одного случая.
---
## Заключение
- Лучшая модель в среднем — Random Forest / XGBoost.
- Возможные улучшения: log-преобразование цели, hyperparameter tuning, стэкинг.
- Приложение готово к интеграции в страховой workflow.
"""


def presentation_page() -> None:
    st.title("🎤 Презентация проекта")
    st.caption(
        "Слайды на streamlit-reveal-slides. Управление — стрелками "
        "клавиатуры или мышью."
    )

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox(
            "Тема",
            AVAILABLE_THEMES,
            index=0,
            help=(
                "Темы поставляются вместе с пакетом streamlit-reveal-slides. "
                "Светлые темы (white, beige, sky, solarized, simple, serif) "
                "лучше читаются для светлых слайдов."
            ),
        )
        height = st.number_input(
            "Высота слайдов (px)",
            min_value=300,
            max_value=1200,
            value=600,
            step=50,
        )
        transition = st.selectbox(
            "Переход между слайдами",
            AVAILABLE_TRANSITIONS,
            index=AVAILABLE_TRANSITIONS.index("slide"),
        )
        selected_plugin_labels = st.multiselect(
            "Плагины Reveal.js",
            options=list(PLUGIN_OPTIONS.keys()),
            default=[
                "Подсветка кода (highlight)",
                "Поиск (search)",
            ],
            help=(
                "Плагины подгружаются при инициализации Reveal.js. "
                "При изменении этого списка слайды пересоздаются заново."
            ),
        )
        st.caption(
            "ℹ️ После изменения темы / перехода / плагинов слайды "
            "автоматически перезагружаются — стрелка вернётся к началу."
        )

    if not REVEAL_AVAILABLE:
        st.warning(
            "Пакет `streamlit-reveal-slides` не установлен — отображаю "
            "презентацию как обычный Markdown.\nУстановите его командой "
            "`pip install streamlit-reveal-slides`."
        )
        st.markdown(PRESENTATION_MARKDOWN)
        return

    plugins = [PLUGIN_OPTIONS[label] for label in selected_plugin_labels]

    # Ключ зависит от всех настроек — при их изменении Streamlit ремонтирует
    # компонент и Reveal.js инициализируется заново. Без этого Reveal.configure()
    # не успевает применить новый список плагинов / перехода.
    component_key = "slides-{theme}-{height}-{transition}-{plugins}".format(
        theme=theme,
        height=int(height),
        transition=transition,
        plugins=",".join(sorted(plugins)) or "none",
    )

    rs.slides(
        PRESENTATION_MARKDOWN,
        height=int(height),
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
            # Дополнительные параметры Reveal, которые приятно иметь:
            "controls": True,
            "progress": True,
            "history": False,
            "center": True,
            "slideNumber": "c/t",
            "transitionSpeed": "default",
        },
        markdown_props={"data-separator-vertical": "^--$"},
        key=component_key,
    )


presentation_page()
