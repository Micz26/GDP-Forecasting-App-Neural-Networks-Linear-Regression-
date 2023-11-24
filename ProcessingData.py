import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.processed_df = None

    def handle_missing_values(self):
        GDP_df_NAN = self.df[self.df.isnull().any(axis=1)]
        self.df.dropna(inplace=True)
        for index, country in GDP_df_NAN.iterrows():
            model = LinearRegression()
            X_train = []
            y_train = []
            X = []
            for year in GDP_df_NAN.columns[1:]:
                if pd.notna(country[year]):
                    X_train.append(int(year))
                    try:
                        GDP = int(country[year])
                        y_train.append(GDP)
                    except ValueError:
                        GDP = country[year].strip().replace(" ", "").lower()
                        if GDP.endswith("k"):
                            GDP = float(GDP[:-1]) * 1000
                        y_train.append(int(GDP))
                else:
                    X.append(int(year))
            X_train = np.array(X_train).reshape(-1, 1)
            y_train = np.array(y_train)
            model.fit(X_train, y_train)
            X = np.array(X).reshape(-1, 1)
            y = model.predict(X)
            predicted_data = {int(year[0]): int(prediction) for year, prediction in zip(X, y)}

            for year in GDP_df_NAN.columns[1:]:
                if not pd.notna(country[year]):
                    GDP_df_NAN.loc[index, year] = predicted_data[int(year)]

        merged_df = pd.concat([self.df, GDP_df_NAN], ignore_index=True)
        return merged_df

    def processing_data(self):
        merged_df = self.handle_missing_values()
        for index, country in merged_df.iterrows():
            for year in merged_df.columns[1:]:
                try:
                    val = int(country[year])
                    if val < 0:
                        merged_df.loc[index, year] = 0
                    else:
                        merged_df.loc[index, year] = val
                except ValueError:
                    val = country[year].strip().replace(" ", "").lower()
                    if val.endswith("k"):
                        val = float(val[:-1]) * 1000
                    merged_df.loc[index, year] = int(val)
        self.processed_df = merged_df


def main():
    GDP_df = pd.read_csv('C:\\Users\\mikol\\PycharmProjects\\GDP Forecasting\\gdppercapita_us_inflation_adjusted.csv')
    data_processor = DataProcessor(GDP_df)
    data_processor.handle_missing_values()
    data_processor.processing_data()
    data_processor.processed_df.to_csv('dppercapita_us_processed')
if __name__ == '__main__':
    main()
