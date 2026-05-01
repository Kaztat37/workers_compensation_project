"""Страница презентации проекта (streamlit-reveal-slides).

Все переключатели в сайдбаре действительно переинициализируют слайды:

- **Тема** — выбирается из списка CSS, которые поставляются с пакетом
  streamlit-reveal-slides (см. ``frontend/build/static/css/``).
- **Высота слайдов** — пересчитывается при ресайзе iframe.
- **Переход** — прописывается **per-slide** через директиву
  ``<!-- .slide: data-transition="X" -->`` в начале каждого слайда.
  Это надёжнее, чем глобальный ``Reveal.configure({transition})``: эта
  встроенная сборка компонента вызывает ``Reveal.initialize(...)`` только
  один раз на mount, поэтому глобальный ``transition`` после первого
  рендера не пересоздаёт CSS-классы у уже отрисованных секций.
- **Плагины** — отображаемые имена мапятся на реальные ключи Reveal
  (``RevealHighlight``, ``RevealSearch``, ``RevealNotes``, ``RevealZoom``,
  ``RevealMath.KaTeX``).

Чтобы изменения переходов и плагинов гарантированно применялись, в
``rs.slides(...)`` передаётся параметр ``key``, зависящий от текущих
настроек: при его изменении Streamlit ремонтирует компонент.

Дополнительно на каждом рендере в начало markdown инжектируется блок
``<style>``: он отключает агрессивный ``text-transform: uppercase`` и
крупный шрифт ``h1`` у тем вроде beige / league / dracula, из-за которых
длинные русские заголовки на 1-м слайде не помещались в кадр.
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

# Виды переходов Reveal.js (передаются per-slide через data-transition).
AVAILABLE_TRANSITIONS = ["none", "fade", "slide", "convex", "concave", "zoom"]
AVAILABLE_TRANSITION_SPEEDS = ["default", "fast", "slow"]

# Плагины Reveal.js, доступные в встроенной сборке streamlit-reveal-slides.
# Ключ — то, что видит пользователь; значение — имя плагина для Reveal.
PLUGIN_OPTIONS: dict[str, str] = {
    "Подсветка кода (highlight)": "RevealHighlight",
    "Поиск (search)": "RevealSearch",
    "Заметки докладчика (notes)": "RevealNotes",
    "Зум по двойному клику (zoom)": "RevealZoom",
    "Формулы KaTeX (math)": "RevealMath.KaTeX",
}


# Содержание презентации в виде списка слайдов. Это нужно, чтобы
# программно дописывать к каждому слайду директиву data-transition,
# которая по факту определяет визуальный эффект перехода.
SLIDES: list[str] = [
    # 1 — титульный
    (
        "# Прогноз стоимости страховых выплат\n"
        "#### Workers Compensation · UltimateIncurredClaimCost"
    ),
    # 2 — введение
    (
        "## Введение\n"
        "- Анализ данных о страховых случаях компенсации работникам.\n"
        "- Цель — предсказать **итоговую** стоимость страхового возмещения.\n"
        "- Датасет: Workers Compensation (OpenML id = 42876, **100 000** записей)."
    ),
    # 3 — бизнес-задача
    (
        "## Бизнес-задача\n"
        "- Страховые компании нуждаются в точной оценке будущих выплат.\n"
        "- Начальная оценка (`InitialCaseEstimate`) часто отличается от итоговой.\n"
        "- Точные прогнозы помогают формировать резервы и тарифицировать риски."
    ),
    # 4 — описание данных
    (
        "## Описание данных\n"
        "- 13 признаков + целевая переменная `UltimateIncurredClaimCost`.\n"
        "- Категориальные: `Gender`, `MaritalStatus`, `PartTimeFullTime`, "
        "`ClaimDescription`.\n"
        "- Datetime: `DateTimeOfAccident`, `DateReported`.\n"
        "- Числовые: возраст, зарплата, часы, начальная оценка и др."
    ),
    # 5 — этапы работы
    (
        "## Этапы работы\n"
        "1. Загрузка данных через `sklearn.datasets.fetch_openml`.\n"
        "2. Предобработка: даты → `AccidentMonth`, `AccidentDayOfWeek`, `ReportingDelay`.\n"
        "3. `LabelEncoder` для категорий, `StandardScaler` для числовых.\n"
        "4. `train_test_split` 80 / 20.\n"
        "5. Обучение Linear / Ridge / Random Forest / XGBoost.\n"
        "6. Оценка MAE, MSE, RMSE, R² и анализ важности признаков."
    ),
    # 6 — пример формулы
    (
        "## Пример формулы для KaTeX\n"
        "Среднеквадратичная ошибка:\n\n"
        r"$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$"
        "\n\n"
        "> 💡 Чтобы формула отрисовалась, включите плагин «Формулы KaTeX (math)»."
    ),
    # 7 — пример кода
    (
        "## Пример блока кода (highlight)\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestRegressor\n\n"
        "model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "preds = model.predict(X_test)\n"
        "```\n"
        "> 💡 Включите плагин «Подсветка кода (highlight)», чтобы код подсветился."
    ),
    # 8 — ключевые признаки
    (
        "## Ключевые признаки\n"
        "- **InitialCaseEstimate** — обычно №1 по важности.\n"
        "- **WeeklyPay**, **Age**, **HoursWorkedPerWeek**.\n"
        "- **ReportingDelay** — задержка отчёта о случае.\n"
        "- **ClaimDescription** — тип травмы."
    ),
    # 9 — сравнение моделей
    (
        "## Сравнение моделей\n\n"
        "| Модель              | Сильные стороны                              |\n"
        "|---------------------|----------------------------------------------|\n"
        "| Linear Regression   | Простота, интерпретируемость                 |\n"
        "| Ridge Regression    | Регуляризация, мультиколлинеарность          |\n"
        "| Random Forest       | Устойчивость, нелинейности, важность фич     |\n"
        "| XGBoost             | Высокая точность, регуляризация              |"
    ),
    # 10 — Streamlit-приложение
    (
        "## Streamlit-приложение\n"
        "- Многостраничное приложение на `st.navigation` + `st.Page`.\n"
        "- Загрузка данных, обучение и метрики «в один клик».\n"
        "- Графики: распределение цели, предсказания vs факт, важность признаков.\n"
        "- Форма для интерактивного предсказания одного случая."
    ),
    # 11 — заключение
    (
        "## Заключение\n"
        "- Лучшая модель в среднем — Random Forest / XGBoost.\n"
        "- Возможные улучшения: log-преобразование цели, hyperparameter tuning,\n"
        "  стэкинг моделей.\n"
        "- Приложение готово к интеграции в страховой workflow."
    ),
]


# CSS-override, который инжектится прямо в первый слайд:
#
# 1. ``.reveal.fade { opacity: 1 !important; }`` — критическое исправление:
#    в iframe streamlit-reveal-slides подгружается bootstrap.min.css, в нём
#    есть утилита-класс ``.fade:not(.show) { opacity: 0; }``. Когда Reveal.js
#    включает fade-переход, он добавляет класс ``fade`` корневому элементу
#    ``.reveal`` — Bootstrap тут же делает его прозрачным, и весь слайд
#    становится чёрным. Селектор ``.reveal.fade`` специфичнее одиночного
#    ``.fade``, плюс ``!important`` гарантированно перебивает Bootstrap.
#
# 2. h1/h2 — нивелируем uppercase + крупный шрифт у «цветных» тем
#    (beige, league, dracula, blood, sky, night, …), чтобы длинные русские
#    заголовки не вылезали за края слайда.
SLIDE_STYLE_BLOCK = """<style>
.reveal.fade { opacity: 1 !important; }
.reveal h1 {
  font-size: 1.9em !important;
  text-transform: none !important;
  word-break: keep-all;
  hyphens: manual;
  line-height: 1.1;
}
.reveal h2 {
  font-size: 1.55em !important;
  text-transform: none !important;
  word-break: keep-all;
  line-height: 1.15;
}
.reveal h3, .reveal h4, .reveal h5 {
  text-transform: none !important;
}
.reveal section {
  font-size: 0.9em;
}
.reveal pre {
  font-size: 0.55em;
}
.reveal pre code {
  max-height: 380px;
}
.reveal table {
  font-size: 0.7em;
  margin: 0 auto;
}
</style>"""


def build_markdown(transition: str, transition_speed: str) -> str:
    """Собирает итоговый markdown.

    На каждом слайде в начало добавляется директива
    ``<!-- .slide: data-transition="X" data-transition-speed="Y" -->`` —
    благодаря этому Reveal.js использует выбранный пользователем эффект
    перехода для конкретной секции (глобальный конфиг применяется только
    на mount, и Streamlit после смены настроек не всегда полностью
    пересоздаёт компонент).

    Директива ставится также на первый слайд, чтобы переход «назад» с
    второго слайда на первый тоже использовал выбранную анимацию (Reveal
    при таком переходе берёт data-transition с **целевой** секции).

    В первый слайд дополнительно мерджится :data:`SLIDE_STYLE_BLOCK` с
    CSS-фиксами (важнее всего — нивелирование bootstrap-овского
    ``.fade:not(.show) { opacity: 0 }``, который иначе делает чёрный
    слайд при fade-переходе). Блок встраивается в первый слайд, а не
    идёт отдельной секцией, чтобы не появлялся пустой слайд в начале.
    """

    transition_directive = (
        f'<!-- .slide: data-transition="{transition}" '
        f'data-transition-speed="{transition_speed}" -->'
    )
    parts: list[str] = []
    for index, slide_md in enumerate(SLIDES):
        if index == 0:
            prefix = f"{SLIDE_STYLE_BLOCK}\n\n{transition_directive}\n\n"
        else:
            prefix = f"{transition_directive}\n\n"
        parts.append(prefix + slide_md.strip())
    return "\n\n---\n\n".join(parts)


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
            help=(
                "Прописывается **в каждый слайд** через директиву "
                "data-transition — поэтому переключение действительно "
                "меняет анимацию (а не только глобальный конфиг Reveal)."
            ),
        )
        transition_speed = st.selectbox(
            "Скорость перехода",
            AVAILABLE_TRANSITION_SPEEDS,
            index=AVAILABLE_TRANSITION_SPEEDS.index("slow"),
            help=(
                "`fast` ≈ 0.2с, `default` ≈ 0.4с, `slow` ≈ 1с. "
                "При коротком переходе fade/zoom могут визуально "
                "восприниматься как простое перелистывание — "
                "включите `slow`, чтобы анимация была заметна."
            ),
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
        st.markdown(build_markdown(transition, transition_speed))
        return

    plugins = [PLUGIN_OPTIONS[label] for label in selected_plugin_labels]

    # Ключ зависит от всех настроек — при их изменении Streamlit ремонтирует
    # компонент и Reveal.js инициализируется заново. Без этого Reveal.configure()
    # не успевает применить новый список плагинов.
    component_key = "slides-{theme}-{height}-{transition}-{speed}-{plugins}".format(
        theme=theme,
        height=int(height),
        transition=transition,
        speed=transition_speed,
        plugins=",".join(sorted(plugins)) or "none",
    )

    rs.slides(
        build_markdown(transition, transition_speed),
        height=int(height),
        theme=theme,
        config={
            # Глобально дублируем выбранный transition, но в конечном счёте
            # перевешивает per-slide data-transition, заданный в build_markdown.
            "transition": transition,
            "transitionSpeed": transition_speed,
            "plugins": plugins,
            "controls": True,
            "progress": True,
            "history": False,
            "center": True,
            "slideNumber": "c/t",
        },
        markdown_props={"data-separator-vertical": "^--$"},
        key=component_key,
    )


presentation_page()
