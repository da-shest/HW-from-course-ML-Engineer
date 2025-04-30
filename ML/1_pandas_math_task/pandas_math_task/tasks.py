import numpy as np
import pandas as pd
from scipy import stats

def task1_numpy_arrays(matrix_file='../data/matrix.npy'):
    """
    Задание 1: Работа с NumPy массивами

    Дана матрица корреляций в файле matrix.npy.
    
    Вычислите:
    1. Сумму всех элементов матрицы (округлите до 2 знаков после запятой)
    2. Среднее значение элементов матрицы (округлите до 2 знаков после запятой)
    3. Максимальное значение в матрице (округлите до 2 знаков после запятой)
    4. Минимальное значение в матрице (округлите до 2 знаков после запятой)

    Returns:
        dict: {
            'sum': сумма, 
            'mean': среднее, 
            'max': максимум, 
            'min': минимум
        }
    """
    # Загрузка данных
    matrix = np.load(matrix_file)
    
    # Ваше решение

    return {
        'sum': matrix.sum().round(2),
        'mean': matrix.mean().round(2),
        'max': matrix.max().round(2),
        'min': matrix.min().round(2)
    } # Верните словарь с результатами


def task2_dataframe_basic(data_file='../data/data_pd.csv'):
    """
    Задание 2: Базовые операции с DataFrame

    Используя DataFrame из сгенерированных данных в data_pd.csv, вычислите:
    1. Средний возраст сотрудников (округлите до 2 знаков после запятой)
    2. Медианный доход (округлите до 2 знаков после запятой)
    3. Количество сотрудников с высшим образованием (PhD)
    4. Максимальный опыт работы (округлите до 2 знаков после запятой)
    5. Среднюю производительность (округлите до 3 знаков после запятой)
    6. Количество сотрудников в IT отделе

    Returns:
        dict: {
            'mean_age': средний_возраст, 
            'median_income': медианный_доход, 
            'phd_count': количество_PhD, 
            'max_experience': максимальный_опыт, 
            'mean_performance': средняя_производительность, 
            'it_count': количество_IT
        }
    """
    # Загрузка данных
    df = pd.read_csv(data_file)
    
    # Ваше решение
    
    return {
        'mean_age': df['age'].mean().round(2),
        'median_income': df['income'].median().round(2),
        'phd_count': (df['education'] == 'PhD').sum(),
        'max_experience': df['experience'].max().round(2),
        'mean_performance': df['performance'].mean().round(3),
        'it_count': (df['department'] == 'IT').sum()
    } # Верните словарь с результатами


def task3_groupby_aggregation(data_file='../data/data_pd.csv'):
    """
    Задание 3: Группировка и агрегация

    Используя DataFrame из data_pd.csv, вычислите:
    1. Средний доход по отделам (округлите до 2 знаков после запятой для каждого отдела)
    2. Максимальный бонус по уровням образования (округлите до 2 знаков после запятой для каждого уровня)
    3. Количество сотрудников по отделам
    4. Средний опыт работы по отделам (округлите до 2 знаков после запятой для каждого отдела)

    Returns:
        dict: {
            'avg_income_by_dept': {
                'IT': средний_доход_IT,
                'HR': средний_доход_HR,
                'Finance': средний_доход_Finance,
                'Marketing': средний_доход_Marketing
            },
            'max_bonus_by_education': {
                'Bachelor': максимальный_бонус_Bachelor,
                'Master': максимальный_бонус_Master,
                'PhD': максимальный_бонус_PhD
            },
            'count_by_dept': {
                'IT': количество_IT,
                'HR': количество_HR,
                'Finance': количество_Finance,
                'Marketing': количество_Marketing
            },
            'avg_exp_by_dept': {
                'IT': средний_опыт_IT,
                'HR': средний_опыт_HR,
                'Finance': средний_опыт_Finance,
                'Marketing': средний_опыт_Marketing
            }
        }
    """
    # Загрузка данных
    df = pd.read_csv(data_file)
    
    # Ваше решение
    
    return {
        'avg_income_by_dept': df.groupby('department')['income'].mean().round(2).to_dict(),
        'max_bonus_by_education': df.groupby('education')['bonus'].max().round(2).to_dict(),
        'count_by_dept': df.groupby('department').size().to_dict(),
        'avg_exp_by_dept': df.groupby('department')['experience'].mean().round(2).to_dict()
    } # Верните словарь с результатами


def task4_data_filtering(data_file='../data/data_pd.csv'):
    """
    Задание 4: Фильтрация данных

    Используя DataFrame из data_pd.csv, вычислите:
    1. Средний доход сотрудников старше 30 лет (округлите до 2 знаков после запятой)
    2. Максимальный бонус в IT отделе (округлите до 2 знаков после запятой)
    3. Средний опыт работы сотрудников с PhD (округлите до 2 знаков после запятой)
    4. Количество сотрудников с доходом выше 100000
    5. Среднюю удовлетворенность в Finance отделе (округлите до 2 знаков после запятой)

    Returns:
        dict: {
            'mean_income_over_30': средний_доход_старше_30,
            'max_bonus_it': максимальный_бонус_IT,
            'mean_exp_phd': средний_опыт_PhD,
            'high_income_count': количество_высокий_доход,
            'mean_satisfaction_finance': средняя_удовлетворенность_Finance
        }
    """
    # Загрузка данных
    df = pd.read_csv(data_file)
    
    # Ваше решение
    
    return {
        'mean_income_over_30': df[df['age'] > 30]['income'].mean().round(2),
        'max_bonus_it': df[df['department'] == 'IT']['bonus'].max().round(2),
        'mean_exp_phd': df[df['education'] == 'PhD']['experience'].mean().round(2),
        'high_income_count': (df['income'] > 100000).sum(),
        'mean_satisfaction_finance': df[df['department'] == 'Finance']['satisfaction'].mean().round(2)
    } # Верните словарь с результатами




def task5_sorting_ranking(data_file='../data/data_pd.csv'):
    """
    Задание 5: Сортировка и ранжирование

    Используя DataFrame из data_pd.csv, вычислите:
    1. Топ-5 сотрудников по доходу (округлите до 2 знаков после запятой)
    2. Топ-5 сотрудников по опыту работы
    3. Топ-5 сотрудников по производительности (округлите до 3 знаков после запятой)
    4. Топ-5 сотрудников по бонусу (округлите до 2 знаков после запятой)

    Returns:
        dict: {
            'top_income': [доход_1, доход_2, доход_3, доход_4, доход_5],
            'top_experience': [опыт_1, опыт_2, опыт_3, опыт_4, опыт_5],
            'top_performance': [производительность_1, производительность_2, производительность_3, производительность_4, производительность_5],
            'top_bonus': [бонус_1, бонус_2, бонус_3, бонус_4, бонус_5]
        }
    """
    # Загрузка данных
    df = pd.read_csv(data_file)
    
    # Ваше решение
    
    return {
        'top_income': df['income'].sort_values(ascending=False).head(5).round(2).to_list(),
        'top_experience': df['experience'].sort_values(ascending=False).head(5).to_list(),
        'top_performance': df['performance'].sort_values(ascending=False).head().round(3).to_list(),
        'top_bonus': df['bonus'].sort_values(ascending=False).head(5).round(2).to_list()
    } # Верните словарь с результатами


def task6_income_statistics(data_file='../data/data_pd.csv'):
    """
    Задание 6: Вычисление статистик

    Используя DataFrame из data_pd.csv, вычислите:
    1. Среднее значение дохода (округлите до 2 знаков после запятой)
    2. Медиану дохода (округлите до 2 знаков после запятой)

    Returns:
        dict: {
            'mean_income': среднее_дохода,
            'median_income': медиана_дохода
        }
    """
    # Загрузка данных
    df = pd.read_csv(data_file)
    
    # Ваше решение
    
    return {
        'mean_income': df['income'].mean().round(2),
        'median_income': df['income'].median().round(2)
    } # Верните словарь с результатами


def task7_bernoulli_distribution(bernoulli_file='../data/bernoulli.npy'):
    """
    Задание 7: Распределение Бернулли

    Дана выборка из распределения Бернулли в файле bernoulli.npy.
    
    Вычислите:
    1. Оценку вероятности успеха p̂ (округлите до 3 знаков после запятой)
    2. Вероятность получить не менее 60 успехов в 100 испытаниях, если вероятность успеха равна p̂ 
       (округлите до 3 знаков после запятой)

    Returns:
        dict: {
            'p_hat': p̂,
            'p_at_least_60': P(X>=60)
        }
    """
    # Загрузка данных
    bernoulli_sample = np.load(bernoulli_file)
    
    # Ваше решение
    p_hat = bernoulli_sample.mean().round(3)
    n = 100
    k = 59
    p_at_least_60 = (1.0 - stats.binom.cdf(k, n, p_hat)).round(3)
    
    return {
        'p_hat': p_hat,
        'p_at_least_60': p_at_least_60
    } # Верните словарь с результатами


def task8_poisson_distribution(poisson_file='../data/poisson.npy'):
    """
    Задание 8: Распределение Пуассона

    Дана выборка из распределения Пуассона в файле poisson.npy.
    
    Вычислите:
    1. Оценку параметра λ (округлите до 2 знаков после запятой)
    2. Вероятность того, что случайная величина примет значение 3 (округлите до 3 знаков после запятой)
    3. Вероятность того, что случайная величина примет значение больше 5 (округлите до 3 знаков после запятой)

    Returns:
        dict: {
            'lambda_hat': λ̂,
            'p_x_equals_3': P(X=3),
            'p_x_greater_5': P(X>5)
        }
    """
    # Загрузка данных
    poisson_sample = np.load(poisson_file)
    
    # Ваше решение
    lambda_hat = poisson_sample.mean().round(2)
    p_x_equals_3 = stats.poisson.pmf(3, mu=lambda_hat).round(3)
    p_x_greater_5 = 1 - stats.poisson.cdf(5, mu=lambda_hat).round(3)
    
    return {
        'lambda_hat': lambda_hat,
        'p_x_equals_3': p_x_equals_3,
        'p_x_greater_5': p_x_greater_5
    } # Верните словарь с результатами


def task9_exponential_distribution(exponential_file='../data/exponential.npy'):
    """
    Задание 9: Экспоненциальное распределение

    Дана выборка из экспоненциального распределения в файле exponential.npy.
    
    Вычислите:
    1. Оценку параметра λ (округлите до 3 знаков после запятой)
    2. Вероятность того, что случайная величина примет значение больше 15 (округлите до 3 знаков после запятой)

    Returns:
        dict: {
            'lambda_hat': λ̂,
            'p_x_greater_15': P(X>15)
        }
    """
    # Загрузка данных
    exp_sample = np.load(exponential_file)
    
    # Ваше решение
    lambda_hat = (1 / exp_sample.mean()).round(3)
    p_x_greater_15 = (1.0 - stats.expon.cdf(15, scale=1 / lambda_hat)).round(3)
    
    return {
        'lambda_hat': lambda_hat,
        'p_x_greater_15': p_x_greater_15
    } # Верните словарь с результатами 