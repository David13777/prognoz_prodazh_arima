import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def load_data(file_path):
    """Загрузка данных из файла Excel"""
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print("Ошибка при загрузке данных:", e)
    return None

def plot_sales_forecast(data, forecast):
    """Визуализация исходных данных и прогноза"""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Продажи'], label='Исходные данные')
    plt.plot(forecast.index, forecast, color='red', linestyle='--', label='Прогноз продаж на год')
    plt.title('Прогноз продаж на год')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_path = 'sales_data.xlsx'

    # Загрузка данных
    data = load_data(file_path)
    if data is None:
        return

    # Преобразование столбца с датами в индекс временного ряда
    data['Дата'] = pd.to_datetime(data['Дата'])
    data.set_index('Дата', inplace=True)

    try:
        # Обучение модели ARIMA
        model = ARIMA(data['Продажи'], order=(5,1,0)) # Примерный порядок (p,d,q), подлежит настройке
        model_fit = model.fit()

        # Прогнозирование продаж на год вперёд
        forecast = model_fit.forecast(steps=12)

        # Вывод графика
        plot_sales_forecast(data, forecast)

    except Exception as e:
        print("Ошибка при обучении модели или построении прогноза:", e)

if __name__ == "__main__":
    main()
