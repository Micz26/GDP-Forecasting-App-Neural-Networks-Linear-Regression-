from NNModel import GdpNnModel
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import sys


class GDPApp(QWidget):
    def __init__(self, file):
        super().__init__()
        self.df = pd.read_csv(file)
        self.model = GdpNnModel(self.df)
        self.model.create_model()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('GDP Prediction App')
        self.layout = QVBoxLayout()

        self.country_input = QLineEdit()
        self.layout.addWidget(self.country_input)

        self.country_button = QPushButton('Predict GDP')
        self.country_button.clicked.connect(self.predict_gdp)
        self.layout.addWidget(self.country_button)

        self.plot_label = QLabel()
        self.layout.addWidget(self.plot_label, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

    def predict_gdp(self):
        country = self.country_input.text()
        plot = self.model.make_predictions(country, 19)
        file_name = f'{country}_gdp_plot.png'
        plt.savefig(file_name)
        plt.close()
        pixmap = QPixmap(file_name).scaled(1040, 780)
        self.plot_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GDPApp('C:\\Users\\mikol\\PycharmProjects\\GDP Forecasting\\gdppercapita_us-processed')
    window.show()
    sys.exit(app.exec_())






