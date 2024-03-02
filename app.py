import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import streamlit as st
import keras


def show_title_with_subtitle():
    # Заголовок и подзаголовок
    st.title("ПРОГНОЗ ВЫЖИВАНИЯ ДЕРЕВЬЕВ.")
    st.write("# Функциональные особенности саженцев деревьев и их способность выживать в окружающей среде.")



def show_info_page():
    st.write("### Задание:")
    st.write(
        " 1. Решить задачу классификации для предоставленных данных с использованием собственной архитектуры полносвязной нейронной сети (используя API Keras фреймворка TensorFlow). \n"
        " 2. Подобрав параметр, обучить модель. \n"
        " 3. Оценить качество модели с помощью отчета о классификации и матрицы путаницы. \n" 
        )
    st.image("https://kfh-fruktovyjsad.ru/images/page/sazhenets.jpg", use_column_width=True)
   
    st.write("### Описание входных данных.")
    st.write(
        "Данные, для которых необходимо получать предсказания, представляют собой подробное признаковое описание факторов, влияющих на выживаемость саженцев:\n"
        " - No - уникальный идентификационный номер саженца.\n"
        " - Species - сорта саженцов (Acer saccharum = 1, Prunus serotina = 2, Quercus alba = 3, Quercus rubra = 4).\n"
        " - Light Cat - категориальный уровень освещенности (Med = 1, Low = 0).\n"
        " - Core - год, когда керн почвы был удален с поля.\n"
        " - Soil - сорта, из которых был взят керн почвы (Prunus serotina = 0, Quercus rubra = 1, Acer rubrum = 2, Populus grandidentata = 3, Sterile = 4, Acer saccharum = 5, Quercus alba 6).\n"
        " - Sterile - была ли почва стерилизована или нет (Non-Sterile = 0, Sterile = 1).\n"
        " - Conspecific - была ли почва конспецифической, гетероспецифичной или стерилизованной конспецифической (Heterospecific = 0, Sterilized = 1, Conspecific = 2).\n"
        " - Census - номер переписи, когда саженец погиб или был собран.\n"
        " - Time - количество дней, в течение которых саженец погиб или был собран.\n"
        " - Event - используется для анализа выживаемости, чтобы указать состояние каждого отдельного саженца в данный момент времени (0 = собран или эксперимент завершен, 1 = мертв).\n"
        )
    st.write("### Введение.")
    st.write(
        " *Полносвязная нейронная сеть* - это тип искусственной нейронной сети, в которой нейроны каждого слоя полностью связаны с нейронами следующего слоя. Это один из простейших типов нейронных сетей, где каждый нейрон входного слоя соединен с каждым нейроном скрытого и выходного слоя.\n"
        " При разработке собственной архитектуры полносвязной нейронной сети для классификации данных, необходимо определить количество слоев и нейронов, используемые активационные функции, метод обучения, другие параметры."
    )
    st.write(
        " *Цель* - создать эффективную нейронную сеть, которая сможет точно классифицировать данные и делать предсказания. " 
    )
    st.write(
        " Для того, чтобы более подробно узнать об API высокого уровня для TensorFlow перейдите по ссылке https://www.tensorflow.org/guide/keras?hl=ru.html."
        )
    st.write("Почему важно поддержание жизнеспособности деревьев:")
    st.video("https://www.youtube.com/watch?v=QH4Pf_jEGE4")
    # st.write("Выполненная работа представляет собой результат участия в соревновании на портале Kaggle. Более подробно"
    #         "ознакомиться с правилами соревнования можно по ссылке ниже:")
    # st.write("https://www.kaggle.com/c/house-prices-advanced-regression-techniques")


def show_predictions_page():
    st.write("Исходные данные для работы: https://drive.google.com/file/d/1V-DA1SLvKphgWi9r3z0o3vZ1k3x0PBj2/view?usp=sharing")
    file = st.file_uploader(label="Выберите csv файл с предобработанными данными для прогнозирования выживания деревьев",
                            type=["csv"],
                            accept_multiple_files=False)
    if file is not None:
        test_data = pd.read_csv(file)
        st.write("### Загруженный файл")
        st.write(test_data)
        make_predictions(get_model(), test_data)


def get_model():
    #return CatBoostRegressor().load_model(os.path.join(os.path.dirname(__file__), "models", "cb.h5"))
    return keras.models.load_model('./models/cb.h5')

#def get_model():
#    model_path = os.path.join(os.path.dirname(__file__), "models", "cb.h5")
#    model = load_model(model_path)
#    return model

# loaded_model = get_model()

#######################################################################################################################
def make_predictions(model, X):
    st.write("### Предсказанные значения")
    # st.markdown("Hello<br />World!")
    pred = pd.DataFrame(model.predict(X))
    st.write(pred)
   ## st.write("### Гистограмма распределения предсказаний")
   ## plot_hist(pred)

def plot_hist(data):
    fig = plt.figure()
    sbn.histplot(data, legend=False)
    st.pyplot(fig)


def select_page():
    # Сайдбар для смены страницы
    return st.sidebar.selectbox("Выберите раздел", ("Главная страница", "Реализация"))


# Стиль для скрытия со страницы меню, футера streamlit и кнопки deploy
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# размещение элементов на странице
show_title_with_subtitle()
st.sidebar.title("Меню")
page = select_page()
st.sidebar.write("☛ **Sledneva Kristina** 2024")

if page == "Главная страница":
    show_info_page()
else:
    show_predictions_page()
    