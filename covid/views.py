import plotly.tools
from django.shortcuts import render
from django.http import JsonResponse

from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt

from covid import df, states, taxa_crescimento_diaria
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from plotly.offline import plot
import pandas as pd
from pmdarima.arima import auto_arima
from fbprophet import Prophet


def index(request):
    context = {'states': states}
    return render(None, 'covidapp.html', context=context)


def get_country(request):
    if request.is_ajax and request.method == 'GET':
        country_name = request.GET.get("country_name")
        country = df.loc[(df.countryregion == country_name) & (df.confirmed > 0)]
        graphs = []
        graphs.append(go.Scatter(x=country['observationdate'], y=country['confirmed'], mode='lines'))
        layout = {
            'title': 'Casos confirmados no Brasil',
            'xaxis_title': 'observationdate',
            'yaxis_title': 'confirmed',
            'height': 420,
            'width': 560,
        }

        # Casos confirmados no Brasil
        fig_01 = px.line(country, 'observationdate', 'confirmed',
                         title=f'Casos confirmados - {country_name}')
        plot_div_01 = plot(fig_01, output_type='div', include_plotlyjs=False)
        # Novos casos por dia no Brasil
        country['novoscasos'] = list(
            map(lambda x: 0 if (x == 0) else country['confirmed'].iloc[x] - country['confirmed'].iloc[x - 1],
                np.arange(country.shape[0])))
        fig_02 = px.line(country, 'observationdate', 'novoscasos', title=f'Novos casos por dia - {country_name}')
        plot_div_02 = plot(fig_02, output_type='div', include_plotlyjs=False)

        # Mortes por COVID-19 no Brasil
        fig_03 = go.Figure()
        fig_03.add_trace(
            go.Scatter(x=country.observationdate, y=country.deaths, name='Mortes',
                       mode="lines+markers", line={'color': 'red'})
        )
        fig_03.update_layout(title=f"Mortes por COVID-19 - {country_name}")
        plot_div_03 = plot(fig_03, output_type='div', include_plotlyjs=False)

        # Taxas de Crescimento diárias
        tx_dia = taxa_crescimento_diaria(country, 'confirmed')
        primeiro_dia = country.observationdate.loc[country.confirmed > 0].min()
        fig_04 = px.line(x=pd.date_range(primeiro_dia, country.observationdate.iloc[-1])[1:],
                         y=tx_dia, title=f"Taxa de crescimento de casos confirmados - {country_name}")
        plot_div_04 = plot(fig_04, output_type='div', include_plotlyjs=False)

        confirmados = country.confirmed
        try:
            confirmados.index = country.observationdate
            res = seasonal_decompose(confirmados)
            fig_05 = make_subplots(rows=4, cols=1)
            fig_05.add_trace(go.Scatter(x=res.observed.index, y=res.observed.values, name='Observed',
                                        mode="lines+markers"), row=1, col=1)
            fig_05.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, name='Trend',
                                        mode="lines+markers"), row=2, col=1)
            fig_05.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, name='Seasonal',
                                        mode="lines+markers"), row=3, col=1)
            fig_05.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, name='Resid',
                                        mode="lines+markers"), row=4, col=1)
            fig_05.add_hline(y=0, line_width=2, line_dash="dash", line_color="black", row=4, col=1)
            fig_05.update_layout(height=1000, title='Seasonal Decomposition')
            plot_div_05 = plot(fig_05, output_type='div', include_plotlyjs=False)
        except:
            b64 = ''
            plot_div_05 = ''

        modelo = auto_arima(confirmados)
        fig_06 = go.Figure(go.Scatter(
            x=confirmados.index, y=confirmados, name='Observados'
        ))
        fig_06.add_trace(go.Scatter(
            x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
        ))
        fig_06.add_trace(go.Scatter(
            x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'
        ))
        fig_06.update_layout(title=f'Previsão de casos confirmados para os próximos 30 dias - {country_name}')

        plot_div_06 = plot(fig_06, output_type='div', include_plotlyjs=False)

        train = confirmados.reset_index()[:-5]
        test = confirmados.reset_index()[-5:]
        train.rename(columns={'observationdate': 'ds', 'confirmed': 'y'}, inplace=True)
        test.rename(columns={'observationdate': 'ds', 'confirmed': 'y'}, inplace=True)

        profeta = Prophet(growth='logistic', changepoint_prior_scale=0.001)
        pop = 1000000
        train['cap'] = pop

        profeta.fit(train)

        future_dates = profeta.make_future_dataframe(periods=200)
        future_dates['cap'] = pop
        forecast = profeta.predict(future_dates)

        fig_07 = go.Figure()

        fig_07.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
        fig_07.add_trace(go.Scatter(x=test.ds, y=test.y, name='Observados - Teste'))
        fig_07.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
        fig_07.update_layout(title=f'Predições de casos confirmados - {country_name} - População: {pop}')
        plot_div_07 = plot(fig_07, output_type='div', include_plotlyjs=False)
        data = {"plot_div_01": plot_div_01,
                "plot_div_02": plot_div_02,
                "plot_div_03": plot_div_03,
                "plot_div_04": plot_div_04,
                "plot_div_05": plot_div_05,
                "plot_div_06": plot_div_06,
                "plot_div_07": plot_div_07,
                }

        return JsonResponse(data, status=200)
    return JsonResponse({}, status=400)
