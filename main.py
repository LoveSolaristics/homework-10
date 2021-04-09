import streamlit as st

import pandas as pd
import numpy as np

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py

from pywaffle import Waffle

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Космос", page_icon='🚀',
)

st.markdown(
    '''
    # Причины неудач космических миссий
    
    ### <center>Разбираемся в том, что влияет на вероятность провала космической миссии</center>
    
    __Содержание:__:
    
    - [Обзор и предобработка данных](#part1)
    - [Анализируем влияние космодрома и его местоположение на космические миссии](#part2)
    - [Анализируем эффект компании на запуск](#part3)
    - [Имеют ли какое-либо влияние ракеты-носители?](#part4)
    - [Время запуска - новое лучше старого?](#part5)
    - [Стоимость космической миссии - Stonks :arrow_up: или :arrow_down:](#part6)
    
    _Якорные ссылки пока не поддерживаются streamlit, но скоро обещали это поправить._
    
    ## Обзор и предобработка данных
    <a id = "part1"></a>
    
    Будем разбирать данные из датасета __All Space Missions from 1957__ ([ссылка]   
    (https://www.kaggle.com/agirlcoding/all-space-missions-from-1957)).
    
    <img src='https://storage.googleapis.com/
    kaggle-datasets-images/828921/1416362/d1834c9d4366150df0ffd5aa2868cd03
    /dataset-cover.jpg?t=2020-08-13-09-37-39' style='max-width: 100%'>
    
    Изначально в датасете 7 информативных колонок:
    - __Company Name__ - еазвание компании, выполнившей запуск
    - __Location__ - место запуска
    - __Datum__ - время и дата запуска
    - __Detail__ - название двигателя ракеты
    - __Status Rocket__ - текущий статус ракеты
    - __Rocket__ - цена в миллионах долларов
    - __Status Mission__ - статус: провал или успех миссии
    ''', unsafe_allow_html=True)

with st.echo(code_location='below'):
    df = pd.read_csv('Space_Corrected.csv')
    df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
    st.write(df)
st.markdown(
    """
    Заведем новую колонку для страны, которая занималась запуском.
    """)

with st.echo(code_location='below'):
    def extract_country(location):
        country = location.split(',')[-1]
        country = country.strip()
        return country


    dict_countries = {
        'Russia': 'Russian Federation',
        "Barents Sea": 'Russian Federation',
        'New Mexico': 'USA',
        "Pacific Missile Range Facility": 'USA',
        "Gran Canaria": 'USA',
        "Yellow Sea": 'China',
        "Shahrud Missile Test Site": "Iran"
    }

    df['Country'] = df['Location'].apply(lambda location: extract_country(location))
    df['Country'] = df['Country'].replace(dict_countries)
st.markdown(
    """
    Преобразуем столбец даты запуска, а также выделим отдельные колонки для года, месяца и дня недели.
    Это понадобится для более удобной работы с данными.
    """)

with st.echo(code_location='below'):
    df['Datum'] = pd.to_datetime(df['Datum'])
    df['Year'] = df['Datum'].apply(lambda datetime: datetime.year)
    df['Month'] = df['Datum'].apply(lambda datetime: datetime.month)
    df['Weekday'] = df['Datum'].apply(lambda datetime: datetime.weekday())
st.markdown(
    """
    Также для удобства обработаем столбец с двигаетелями.
    """)

with st.echo(code_location='below'):
    def getVehicles(detail):
        list_vehicles = []
        list_detail = [x.strip() for x in detail.split('|')]
        for ele in list_detail:
            if 'Atlas' in ele:
                list_vehicles.append('Atlas')
            elif 'Ariane' in ele:
                list_vehicles.append('Ariane')
            elif 'Cosmos' in ele:
                list_vehicles.append('Cosmos')
            elif 'Delta' in ele:
                list_vehicles.append('Delta')
            elif 'Falcon' in ele:
                list_vehicles.append('Falcon')
            elif 'GSLV' in ele:
                list_vehicles.append('GSLV')
            elif 'Long March' in ele:
                list_vehicles.append('Long March')
            elif 'Molniya' in ele:
                list_vehicles.append('Molniya')
            elif 'PSLV' in ele:
                list_vehicles.append('PSLV')
            elif 'Soyuz' in ele:
                list_vehicles.append('Soyuz')
            elif 'Thor' in ele:
                list_vehicles.append('Thor')
            elif 'Titan' in ele:
                list_vehicles.append('Titan')
            elif 'Tsyklon' in ele:
                list_vehicles.append('Tsyklon')
            elif 'Vostok' in ele:
                list_vehicles.append('Vostok')
            elif 'Zenit' in ele:
                list_vehicles.append('Zenit')
            else:
                list_vehicles.append('Other')
        return list_vehicles


    df['Launch Vehicles'] = df['Detail'].apply(lambda x: getVehicles(x))
st.markdown(
    """
    Выведем итоговые данные, чтобы было понятно, что мы сделали с изначальной таблицей
    """)
with st.echo(code_location='below'):
    st.write(df.head())
st.markdown(
    """
    Поскольку мы анализируем причину, 
    по которой некоторые космические миссии терпят неудачу, 
    давайте сначала посмотрим на распределение статусных миссий.
    
    _Для построения графика использовалась библиотека pywaffle. 
    Но график, построенный с её помощью некорректно отображается, поэтому создается фигура,
    сохраняется картинкой, а только затем отображается на странице._
    """)

with st.echo(code_location='below'):
    plt.rcParams['figure.figsize'] = (7, 11)
    plt.rcParams['axes.facecolor'] = '#F0F2F6'
    data = dict(df['Status Mission'].value_counts(normalize=True) * 100)
    fig = plt.figure(
        FigureClass=Waffle,
        columns=10,
        values=data,
        colors=("MediumSpringGreen", "Tomato", "#ff9d3b", "#ffff3b"),
        title={'label': 'Статус миссии', 'loc': 'center'},
        icons='space-shuttle',
        icon_size=20,
        labels=[f"{k} {v:.2f}%" for k, v in data.items()],
        legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'ncol': len(data)},
    )

    plt.savefig('images/WafflePlot.jpg', orientation='landscape',
                format='jpg',
                progressive=True,
                bbox_inches='tight')
    st.image('images/WafflePlot.jpg', use_column_width=True)
st.markdown(
    """
    Мы видим, что большое количество (89,71%) космических миссий являются успешными, 
    а 7,84% миссий - неудачными. Эти 7,84% случаев являются наиболее важными для нашего анализа.
    """)

st.markdown("""
## Анализируем влияние космодрома и его местоположения на космические миссии
<a id='part-2' name='part-2'></a>
""", unsafe_allow_html=True)

st.image('images/Here.jpeg', caption='Местоположение любого запуска ракеты', use_column_width=True)

st.markdown(
    """
    Для начала посмотрим, сколько запусков ракетоносителей производилось каждой из стран. 
    Оценить суммарное количество запусков можно после запуска анимации ниже.
    
    _Предобработка данных для оценки суммарного количества запусков производилась заранее.
    И её результат был записан в отдельную csv таблицу_
    """)

with st.echo(code_location='below'):
    total_launches = pd.read_csv('Total_Launch.csv')
    fig = px.bar(total_launches, x="Country", y="Cummulative_Launches", color="Country",
                 animation_group="Country", animation_frame="Year", range_y=[0, 2200])
    fig.update_layout(
        title={
            'text': "Количество запусков ракотоносителей",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="",
        yaxis_title="Количество запусков")
    st.write(fig)
st.markdown(
    """
    Мы видим, что большое количество космических миссий запускается из России и США. 
    Во многом это произошло из-за космической гонки между двумя странами.
    О ней можно почитать на [Википедии](https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%81%D0%BC%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%B3%D0%BE%D0%BD%D0%BA%D0%B0). 
    Приведу короткую выжимку оттуда:
    
    > Космическая гонка (англ. Space Race) — напряжённое соперничество в области освоения космоса между 
    СССР и США в период с 1957 по 1988 годы. В число событий гонки входят запуски искусственных 
    спутников, полёты в космос животных и человека, а также высадка на Луну. 
    Побочный эффект холодной войны.
    
    > Термин получил своё название по аналогии с гонкой вооружений. Космическая гонка стала важной 
    частью культурного, технологического и идеологического противостояния между СССР и США в 
    период холодной войны. Это было обусловлено тем, что космические исследования имели не только большое
     значение для научных и военных разработок, но и заметный пропагандистский эффект.
     
     
    Теперь давайте посмотрим, насколько успешны и неудачны космические миссии для каждой из этих стран, 
    исходя из предположения, что место запуска относится к стране, которая стояла за космической миссией,
     то есть, если местом запуска является Япония, то будем считать, что всей организационной частью, 
     связанной с миссией, занималась Япония.
    """)

with st.echo(code_location='below'):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(df['Status Mission'])
    colors = {0: 'Tomato', 1: 'DarkOrange', 2: 'Plum', 3: 'DeepSkyBlue'}

    fig = make_subplots(rows=8, cols=2, subplot_titles=df['Country'].unique())
    for i, country in enumerate(df['Country'].unique()):
        counts = df[df['Country'] == country]['Status Mission'].value_counts(normalize=True) * 100
        color = [colors[x] for x in encoder.transform(counts.index)]
        trace = go.Bar(x=counts.index, y=counts.values, name=country, showlegend=False,
                       marker={'color': color})
        fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)

    fig.update_layout(template='gridon',
                      margin=dict(l=80, r=80, t=50, b=50),
                      title={'text': 'Статус миссий (по странам)', 'x': 0.5},
                      height=1500,
                      width=700)
    for i in range(1, 9):
        fig.update_yaxes(title_text='Процентиль', row=i, col=1)

    st.write(fig)
st.markdown(
    """
    __Касаемо успешности запусков__:
    - 🇰🇪 Кения занимает первое место со стопроцентной успешностью (правда там всего 9 запусков).
    - 🇫🇷 Франция, выполнившая 303 космических полета, занимает второе место с процентом успеха 94%.
    - 🇷🇺 Россия, выполнившая 1398 космических миссий, занимает третье место с показателем успеха 93,34%. 
    По сравнению с запусками, проводимыми в :us: США, в России дела обстоят лучше, 
    поскольку у миссий в США показатель успеха составляет около 88%.
    
    __С точки зрения частоты отказов__:
    - 🇧🇷 Бразилия и 🇰🇷 Южная Корея имеют одинаковую частоту отказов - 66,67%. Две трети
     их космических миссий терпят неудачу.
        * Следует отметить, что у Южной Кореи показатель успешности составляет 33%, в то время как 
        Бразилия еще не совершила ни одну успешную космическую миссию.
        * Это не прямо-таки ужасные результаты, поскольку и Южная Корея, и Бразилия предприняли всего 
        3 попытки космического полета.
    - Далее, у нас есть 🇰🇵 Северная Корея с 60% неудач в своих 5 космических полетах.
    - Уровень отказов в 🇮🇷 Иране составляет около 57%. Однако он совершил всего 14 космических полетов.
    """)

st.markdown("""
            ## Анализируем эффект компании на запуск
            <a id='part-3'></a>
            """, unsafe_allow_html=True)

st.image('images/Virgin.png', use_column_width=True,
         caption='Новый корабль от компании Virgin, анонсированный неделю назад')
st.markdown(
    """
    Разобъем наши данные на две таблички - с успешными запусками и не очень. 
    Затем вычислим процент успешности запусков  каждой компании и выведем всё на график.
    """)

with st.echo(code_location='below'):
    SuccessPercentile = df[df['Status Mission'] == 'Success'].groupby('Company Name')[
        'Status Mission'].count()
    for company in SuccessPercentile.index:
        SuccessPercentile[company] = (SuccessPercentile[company] / len(
            df[df['Company Name'] == company])) * 100
    SuccessPercentile = SuccessPercentile.sort_index()

    FailurePercentile = df[df['Status Mission'] == 'Failure'].groupby('Company Name')[
        'Status Mission'].count()
    for company in FailurePercentile.index:
        FailurePercentile[company] = (FailurePercentile[company] / len(
            df[df['Company Name'] == company])) * 100
    FailurePercentile = FailurePercentile.sort_index()

    FailurePercentile = df[df['Status Mission'] == 'Failure'].groupby('Company Name')[
        'Status Mission'].count()
    for company in FailurePercentile.index:
        FailurePercentile[company] = (FailurePercentile[company] / len(
            df[df['Company Name'] == company])) * 100
    FailurePercentile = FailurePercentile.sort_index()

    trace1 = go.Bar(x=SuccessPercentile.index, y=SuccessPercentile.values,
                    name='Процент успешных запусков',
                    opacity=0.7)
    trace2 = go.Bar(x=FailurePercentile.index, y=FailurePercentile.values,
                    name='Процент неудачных запусков',
                    opacity=0.7, visible='legendonly')

    fig = go.Figure([trace1, trace2])
    fig.update_layout(template='gridon', margin=dict(l=80, r=80, t=100, b=100),
                      barmode='stack',
                      title={'text': 'Насколько успешны запуски каждой компании', 'x': 0.5},
                      width=750, yaxis_title='Процентиль', xaxis_title='',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center",
                                  x=0.5))

    st.write(fig)
st.markdown(
    """
    Как мы видим, некоторые компании имеют идеальный процент успеха: 
    ASI, Blue Origin, Douglas, IRGC, Khrunichev, OKB-586,Starsem, Yuzhmash и i-Space.
    
    Некоторые компании напротив - не имеют на счету ни одного успешного запуска: EER, Landspace, 
    OneSpace, Sandia и Virgin Orbit. 
    
    В контексте недавней новости о том, что компания Virgin собирается проводить гражданские полеты 
    в космос на новом судне - пожелаю удачи пассажирам их ракет :angel:
    
    Также можно посмотреть на то, как соотносятся компании и страны, чтобы понимать,
    компании каких стран успешнее/неудачнее проводят свои полеты.
    """)
with st.echo(code_location='below'):
    fig = px.treemap(df, path=['Status Mission', 'Country', 'Company Name'])
    fig.update_layout(template='gridon', margin=dict(l=80, r=80, t=50, b=10),
                      title={'text': 'Статус миссии: страны и компании', 'x': 0.5})
    st.write(fig)
st.markdown(
    """
    Казахстан фигурирует на графике, поскольку Байконур, хоть и находится в аренде у России до 2050 года,
    но находится на территории республики.
    """)

st.markdown("""
            ## Имеют ли какое-либо влияние ракеты-носители?
            <a id='part-4'></a>
            """, unsafe_allow_html=True)

st.image('images/Vesicles.jpeg', use_column_width=True)
st.markdown(
    """
    Сосредоточимся лишь на наиболее распространенных типах двигателей, а остальные запихнем в Other. 
    Можно заметить, что наибольшее количество запусков происходило с двигателями Космос, Молния и Atlas.
    Первые две семьи двигателей принадлежат СССР, а Atlas - США. Но соотношение по странам запуска
    выглядит несколько иначе, чем соотношение количества двигателей. Так произошло потому,
    что во-первых, США использует не только американские двигатели для запуска ракет, а во-вторых,
    Россия экспортирует довольно много ракетных двигателей.
    """)

with st.echo(code_location='below'):
    details = []
    for detail in df.Detail.values:
        d = [x.strip() for x in detail.split('|')]
        for ele in d:
            if 'Atlas' in ele:
                details.append('Atlas')
            elif 'Ariane' in ele:
                details.append('Ariane')
            elif 'Cosmos' in ele:
                details.append('Cosmos')
            elif 'Delta' in ele:
                details.append('Delta')
            elif 'Falcon' in ele:
                details.append('Falcon')
            elif 'GSLV' in ele:
                details.append('GSLV')
            elif 'Long March' in ele:
                details.append('Long March')
            elif 'Molniya' in ele:
                details.append('Molniya')
            elif 'PSLV' in ele:
                details.append('PSLV')
            elif 'Soyuz' in ele:
                details.append('Soyuz')
            elif 'Thor' in ele:
                details.append('Thor')
            elif 'Titan' in ele:
                details.append('Titan')
            elif 'Tsyklon' in ele:
                details.append('Tsyklon')
            elif 'Vostok' in ele:
                details.append('Vostok')
            elif 'Zenit' in ele:
                details.append('Zenit')
            else:
                details.append('Other')

    counts = dict(pd.Series(details).value_counts(sort=True))
    counts = pd.DataFrame(counts, index=['Count']).transpose()

    fig = px.pie(counts, values='Count', names=counts.index)
    st.write(fig)

st.markdown("""
            ## Время запуска - новое лучше старого?
            <a id='part-5'></a>
            """, unsafe_allow_html=True)

st.image('images/launch.jpeg', caption='Запуск ракеты SpaceX Falcon 9 в апреле  2020',
         use_column_width=True)

st.markdown(
    """
    Построим диаграммы, отражающие проент провалов за тот или иной промежуток времени.
    """)

with st.echo(code_location='below'):
    fig = make_subplots(rows=3, cols=1)
    for i, period in enumerate(['Year', 'Month', 'Weekday']):
        data = df[df['Status Mission'] == 'Failure'][period].value_counts().sort_index()
        data = dict((data / df[period].value_counts().sort_index()) * 100.0)
        mean = sum(data.values()) / len(data)
        if period == 'Year':
            x = list(data.keys())
        elif period == 'Month':
            x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь',
                 'Осктябрь', 'Ноябрь', 'Декабрь']
        else:
            x = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']

        trace1 = go.Scatter(x=x, y=list(data.values()), mode='lines', text=list(data.keys()),
                            name=f'Провалы за {period} (англ.)', connectgaps=False)
        trace2 = go.Scatter(x=x, y=[mean] * len(data), mode='lines', showlegend=False,
                            name=f'Среднее количество провалов за {period}',
                            line={'dash': 'dash', 'color':
                                'grey'})
        fig.append_trace(trace1, row=i + 1, col=1)
        fig.append_trace(trace2, row=i + 1, col=1)
    fig.update_layout(template='gridon', height=600,
                      title={
                          'text': 'Проваленные миссии за разные периоды времени',
                          'x': 0.5})
    for i in range(1, 4):
        fig.update_yaxes(title_text='Процентиль', row=i, col=1)
    st.write(fig)
st.markdown(
    """
    Хорошо заметно, что количество провалов с годами упало. Вероятно это связано с развитием технологий.
    
    Также заметно, что запуски в выходные или в понедельник не так успешны, как, например, в среду. 
    Суеверным людям на заметку.
    
    Время года или месяц не дают каких-то интересных результатов и закономерностей. 
    Разве что стоит отметить, что в декабре запускать ракеты безопаснее всего.
    """)

st.markdown("""
            ## Стоимость космической миссии 
            <a id='part-5'></a>
            """, unsafe_allow_html=True)

st.image('images/cost.png', caption='Стоимость мечты всегда высока', use_column_width=True)
st.markdown(
    """
    Очевидно, что стоимось запусков должна снижаться с годами в связи с развитием технологий.
    Однако большое искажение в данные вносит космическая миссия СССР в 1987 году.
    Тогда в запуск вложили более 500 миллионов долларов.
    """)

with st.echo(code_location='below'):
    df.rename(columns={" Rocket": "Rocket"}, inplace=True)
    df['Rocket'] = df['Rocket'].apply(lambda x: str(x).replace(',', ''))
    df['Rocket'] = df['Rocket'].astype('float64')
    df['Rocket'] = df['Rocket'].fillna(0)

    costDict = dict(df[df['Rocket'] > 0].groupby('Year')['Rocket'].mean())
    fig = go.Figure(
        go.Scatter(x=list(costDict.keys()), y=list(costDict.values()), yaxis='y2', mode='lines',
                   showlegend=False, name='Стоимость миссий за год'))
    fig.update_layout(template='gridon', margin=dict(l=80, r=80, t=50, b=50),
                      title={'text': 'Средняя стоимость', 'x': 0.5},
                      yaxis_title='Стоимость (млн. $)', xaxis_title='Год запуска')
    st.write(fig)
st.markdown(
    """
    Рассмотрим, как менялась стоимость миссий по конкретным компаниям.
    
    На самом деле здесь сохраняется общий тренд на снижение стоимости запусков.  Это можно заметить 
    хотя бы по стоимости запусков NASA.
    """)

with st.echo(code_location='below'):
    fig = px.scatter(df[df['Rocket'].between(1, 4999)], x='Year', y='Company Name',
                     color='Status Mission', size='Rocket', size_max=30)
    fig.update_layout(template='gridon', margin=dict(l=80, r=80, t=50, b=50),
                      title={'text': 'Стоимость запусков (по компаниям)',
                             'x': 0.5}, height=650, yaxis_title='',
                      xaxis_title='Год запуска')
    st.write(fig)
st.markdown(
    """
    Также можно отслежить, что ранние запуски ракет SpaceX были неудачными, и у них была заметно более
    низкая стоимость миссии по сравнению с их более поздними космическими миссиями. 
    Таким образом, увеличение бюджета, выделяемого на каждую космическую миссию, 
    помогло им стать более успешными.
    """)
