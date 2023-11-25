from NNModel import GdpNnModel
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\mikol\\OneDrive\\Pulpit\\GDP Forecasting\\gdppercapita_us_processed.csv')
    model = GdpNnModel(df)
    model.create_model()
    model.create_charts()






