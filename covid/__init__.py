import requests
import pandas as pd
import numpy as np
import re

url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 "
                  "Safari/537.36", "X-Requested-With": "XMLHttpRequest"
}


def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()


def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    passado = data.loc[data.observationdate == data_inicio, variable].to_numpy()[0]
    presente = data.loc[data.observationdate == data_fim, variable].to_numpy()[0]

    n = (data_fim - data_inicio).days
    taxa = (presente / passado) ** (1 / n) - 1
    return taxa * 100


def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    data_fim = data.observationdate.iloc[-1]
    n = (data_fim - data_inicio).days
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
        range(1, n + 1)
    ))
    return np.array(taxas) * 100


with requests.get(url, headers=headers) as resp:
    if resp.status_code == 200:
        df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])

df.columns = [corrige_colunas(col) for col in df.columns]

states = df['countryregion'].drop_duplicates().sort_values()
states = states[2:-1]
