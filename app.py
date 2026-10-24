import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import metrics

st.set_page_config(
    page_title="Сервис оценки качества вин",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    model_path = BASE_DIR / 'model.joblib'
    if not model_path.exists():
        st.error(f"Файл модели не найден по пути: {model_path}")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_config():
    config_path = BASE_DIR / 'config.json'
    if config_path.exists():
        return json.loads(config_path.read_text(encoding='utf-8'))
    return {'threshold': 0.268}

@st.cache_data
def load_test_data():
    test_path = BASE_DIR / 'test.csv'
    if not test_path.exists():
        st.error(f"Тестовый датасет test.csv не найден по пути: {test_path}")
        st.stop()
    return pd.read_csv(test_path)

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
config = load_config()
default_threshold = float(config.get('threshold', 0.268))
df_test = load_test_data()

FEATURE_COLS = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'is_red'
]

X_test_data = df_test[FEATURE_COLS]
y_test_data = df_test['target']
p_test_cached = test_proba(model, X_test_data)

st.sidebar.header("Управление порогом")
st.sidebar.markdown(f"Порог по умолчанию из `config.json`: **`{default_threshold:.3f}`**")

current_t = st.sidebar.slider(
    "Порог классификации (t)",
    min_value=0.01,
    max_value=0.99,
    value=default_threshold,
    step=0.01,
    help="Если вероятность хорошего вина >= t, модель относит его к классу 1."
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **О приложении:**
    Модель обучена классифицировать вина на хорошие (target=1) и обычные (target=0).
    Сайдбар задает рабочий порог отсечения для всех вкладок.
    """
)


st.title("Демо-сервис для сомелье - отбор хороших вин")

tab1, tab2, tab3, tab4 = st.tabs([
    "Оценка одного вина",
    "Модельные метрики",
    "Пакетное предсказание",
    "Что выучила модель"
])

with tab1:
    st.header("Оценка одного вина по признакам")
    st.write(
        "Задайте параметры образца. Диапазоны и значения по умолчанию взяты из тестовой выборки (`test.csv`)."
    )
    
    col_wine_type, _ = st.columns([2, 2])
    with col_wine_type:
        wine_type = st.radio("Тип вина", ["Белое вино", "Красное вино"], horizontal=True)
        is_red = 1 if wine_type == "Красное вино" else 0

    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input(
            "Fixed acidity (г/дм³)",
            float(df_test['fixed_acidity'].min()),
            float(df_test['fixed_acidity'].max()),
            float(df_test['fixed_acidity'].median()),
            step=0.1
        )
        volatile_acidity = st.number_input(
            "Volatile acidity (г/дм³)",
            float(df_test['volatile_acidity'].min()),
            float(df_test['volatile_acidity'].max()),
            float(df_test['volatile_acidity'].median()),
            step=0.01
        )
        citric_acid = st.number_input(
            "Citric acid (г/дм³)",
            float(df_test['citric_acid'].min()),
            float(df_test['citric_acid'].max()),
            float(df_test['citric_acid'].median()),
            step=0.01
        )
        residual_sugar = st.number_input(
            "Residual sugar (г/дм³)",
            float(df_test['residual_sugar'].min()),
            float(df_test['residual_sugar'].max()),
            float(df_test['residual_sugar'].median()),
            step=0.1
        )

    with col2:
        chlorides = st.number_input(
            "Chlorides (г/дм³)",
            float(df_test['chlorides'].min()),
            float(df_test['chlorides'].max()),
            float(df_test['chlorides'].median()),
            step=0.001,
            format="%.3f"
        )
        free_sulfur_dioxide = st.number_input(
            "Free SO₂ (мг/дм³)",
            float(df_test['free_sulfur_dioxide'].min()),
            float(df_test['free_sulfur_dioxide'].max()),
            float(df_test['free_sulfur_dioxide'].median()),
            step=1.0
        )
        total_sulfur_dioxide = st.number_input(
            "Total SO₂ (мг/дм³)",
            float(df_test['total_sulfur_dioxide'].min()),
            float(df_test['total_sulfur_dioxide'].max()),
            float(df_test['total_sulfur_dioxide'].median()),
            step=1.0
        )
        density = st.number_input(
            "Density (г/см³)",
            float(df_test['density'].min()),
            float(df_test['density'].max()),
            float(df_test['density'].median()),
            step=0.0005,
            format="%.4f"
        )

    with col3:
        pH = st.number_input(
            "pH",
            float(df_test['pH'].min()),
            float(df_test['pH'].max()),
            float(df_test['pH'].median()),
            step=0.01
        )
        sulphates = st.number_input(
            "Sulphates (г/дм³)",
            float(df_test['sulphates'].min()),
            float(df_test['sulphates'].max()),
            float(df_test['sulphates'].median()),
            step=0.01
        )
        alcohol = st.number_input(
            "Alcohol (% об.)",
            float(df_test['alcohol'].min()),
            float(df_test['alcohol'].max()),
            float(df_test['alcohol'].median()),
            step=0.1
        )

    input_sample = pd.DataFrame([{
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': volatile_acidity,
        'citric_acid': citric_acid,
        'residual_sugar': residual_sugar,
        'chlorides': chlorides,
        'free_sulfur_dioxide': free_sulfur_dioxide,
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'is_red': is_red
    }])

    prob_single = model.predict_proba(input_sample)[0, 1]
    is_good = prob_single >= current_t

    st.divider()
    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric(
            label="Вероятность хорошего вина",
            value=f"{prob_single * 100:.1f}%",
            delta=f"{(prob_single - current_t) * 100:+.1f}% от порога {current_t:.2f}"
        )
    with res2:
        if is_good:
            st.success(f"**Вердикт: ХОРОШЕЕ ВИНО**\n\nВероятность ({prob_single:.3f}) выше порога ({current_t:.3f}).")
        else:
            st.warning(f"**Вердикт: ОБЫЧНОЕ ВИНО**\n\nВероятность ({prob_single:.3f}) ниже установленного порога ({current_t:.3f}).")


with tab2:
    st.header("Модельные метрики на тестовой выборке")
    st.write(
        f"Текущий порог классификации: **`t = {current_t:.3f}`**."
    )

    y_pred_curr = (p_test_cached >= current_t).astype(int)

    acc = metrics.accuracy_score(y_test_data, y_pred_curr)
    prec = metrics.precision_score(y_test_data, y_pred_curr, zero_division=0)
    rec = metrics.recall_score(y_test_data, y_pred_curr, zero_division=0)
    f1 = metrics.f1_score(y_test_data, y_pred_curr, zero_division=0)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Precision", f"{prec:.4f}")
    m_col2.metric("Recall", f"{rec:.4f}")
    m_col3.metric("F1-score", f"{f1:.4f}")
    m_col4.metric("Accuracy", f"{acc:.4f}")

    fig_col1, fig_col2, fig_col3 = st.columns(3)

    with fig_col1:
        cm = metrics.confusion_matrix(y_test_data, y_pred_curr)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
            xticklabels=['Обычное (0)', 'Хорошее (1)'],
            yticklabels=['Обычное (0)', 'Хорошее (1)']
        )
        ax_cm.set_title(f"Матрица ошибок (t = {current_t:.2f})", fontsize=11)
        ax_cm.set_xlabel("Предсказание")
        ax_cm.set_ylabel("Истина")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with fig_col2:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_test_data, p_test_cached)
        roc_auc = metrics.roc_auc_score(y_test_data, p_test_cached)
        
        tn_val, fp_val, fn_val, tp_val = cm.ravel()
        curr_fpr = fp_val / (fp_val + tn_val) if (fp_val + tn_val) > 0 else 0
        curr_tpr = rec

        fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
        ax_roc.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], linestyle=':', color='gray')
        ax_roc.scatter([curr_fpr], [curr_tpr], color='red', s=80, zorder=5, label=f'Порог {current_t:.2f}')
        ax_roc.set_title("ROC-кривая", fontsize=11)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=8, loc='lower right')
        ax_roc.grid(True)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with fig_col3:
        p_curve, r_curve, _ = metrics.precision_recall_curve(y_test_data, p_test_cached)
        pr_auc = metrics.average_precision_score(y_test_data, p_test_cached)

        fig_pr, ax_pr = plt.subplots(figsize=(4, 3.5))
        ax_pr.plot(r_curve, p_curve, color='green', label=f'PR (AUC = {pr_auc:.3f})')
        ax_pr.axhline(y=y_test_data.mean(), linestyle=':', color='gray', label=f'Base ({y_test_data.mean():.2f})')
        ax_pr.scatter([rec], [prec], color='red', s=80, zorder=5, label=f'Порог {current_t:.2f}')
        ax_pr.set_title("PR-кривая", fontsize=11)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8, loc='lower left')
        ax_pr.grid(True)
        plt.tight_layout()
        st.pyplot(fig_pr)
        plt.close(fig_pr)

    st.divider()
    
with tab3:
    st.header("Пакетная оценка партии вин из CSV-файла")
    st.write(
        f"Загрузите CSV-файл с анализами. "
        f"Оценка будет произведена по порогу из сайдбара: **`t = {current_t:.3f}`**."
    )

    uploaded_file = st.file_uploader("Загрузить файл вин (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            
            normalized_cols = {c: c.strip().replace(' ', '_') for c in batch_raw.columns}
            batch_df = batch_raw.rename(columns=normalized_cols)
            
            missing = [col for col in FEATURE_COLS if col not in batch_df.columns]

            if missing:
                st.error(
                    f"Ошибка валидации - в загруженном файле отсутствуют обязательные столбцы ({len(missing)} шт.):\n\n"
                    f"`{missing}`\n\n"
                    f"Файл должен содержать все 12 признаков: `{FEATURE_COLS}`."
                )
            else:
                st.success(f"Файл успешно прочитан - всего строк: {len(batch_df)}.")
                
                features_only = batch_df[FEATURE_COLS]
                probs = model.predict_proba(features_only)[:, 1]
                preds = (probs >= current_t).astype(int)

                result_df = batch_df.copy()
                result_df['prob_good_wine'] = probs.round(4)
                result_df['decision'] = preds
                result_df['verdict'] = np.where(preds == 1, 'Хорошее', 'Обычное')

                good_count = int(preds.sum())
                st.info(
                    f"Результат отбора: **{good_count}** из **{len(batch_df)}** вин "
                    f"({good_count / len(batch_df) * 100:.1f}%) рекомендованы к включению в подборку."
                )

                st.dataframe(result_df.head(100), use_container_width=True)

                csv_output = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать CSV с решениями модели",
                    data=csv_output,
                    file_name="wine_recommendations.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Не удалось обработать CSV-файл. Ошибка: {e}")
    else:
        st.info("Скачайте пример тестовой партии вин без колонки target для проверки:")
        sample_batch = X_test_data.head(25).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать образец партии (25 вин)",
            data=sample_batch,
            file_name="sample_batch.csv",
            mime="text/csv"
        )

with tab4:
    st.header("Анализ весов обученной модели")
    st.write(
        "Коэффициенты логистической регрессии для стандартизированных признаков. "
    )

    lr_model = model.named_steps['logisticregression']
    weights = lr_model.coef_[0]

    df_coefs = pd.DataFrame({
        'Признак': FEATURE_COLS,
        'Вес': weights
    }).sort_values(by='Вес', ascending=True)

    fig_w, ax_w = plt.subplots(figsize=(10, 6))
    bar_colors = ['#2ca02c' if w > 0 else '#d62728' for w in df_coefs['Вес']]
    
    bars = ax_w.barh(df_coefs['Признак'], df_coefs['Вес'], color=bar_colors, alpha=0.85)
    ax_w.axvline(0, color='black', linewidth=1, linestyle='--')
    ax_w.set_title("Веса признаков логистической регрессии", fontsize=13)
    ax_w.set_xlabel("Величина коэффициента", fontsize=11)
    ax_w.grid(True, axis='x', linestyle=':', alpha=0.6)

    for bar in bars:
        w_val = bar.get_width()
        ha_align = 'left' if w_val >= 0 else 'right'
        offset = 0.02 if w_val >= 0 else -0.02
        ax_w.text(w_val + offset, bar.get_y() + bar.get_height() / 2, f"{w_val:+.2f}", va='center', ha=ha_align, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_w)
    plt.close(fig_w)
