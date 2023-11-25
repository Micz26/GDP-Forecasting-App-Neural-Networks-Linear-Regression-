# GDP-Forecasting-app-Neural-Networks-Linear-Regression-
Website hosted on AWS amplify - [link](https://dev.d2ueoghao3mx1q.amplifyapp.com/) that predicts future GDP per capita based on previous values for years 1960-2021. It creates and displays charts (GDP/years). It has been developed using neural networks and linear regression. It utilizes AWS S3 to store pregenerated GDP plots for every country and AWS Lambda alongside API Gateway to handle the application's core logic.

- [Overview](#overview)
    - Data processing
    - Prediction model
    - App Gui
    - Cloud Architecture
- [Data analysis](#data-analysis)
- [License](#license)
- [Contact](#contact)



## Overview
The dataset [gdppercapita_us_inflation_adjusted.csv](GUI/gdppercapita_us_inflation_adjusted.csv) was found as a CSV file on https://www.kaggle.com/datasets. It contained data on the GDP of 211 countries over the years 1960-2021. However, the dataset had many missing values, so it required preprocessing. Once I had a clean and ready-to-use dataframe, I created a deep learning neural network to predict future GDP per capita values.

### Data processing
To process the data, I used the Linear Regression model from the scikit-learn library. For each country in the dataframe, I created a separate Linear Regression model, trained using the available data from years without missing values. With this model, I could predict the missing GDP values for each year.

However, the models occasionally produced negative predictions. Since GDP cannot be negative, I implemented a correction step, setting any negative predicted values to 0. Final CSV file can be found [here](GUI/gdppercapita_us-processed.csv).

### Prediction model
The Neural Network model consists of three hidden layers with 32, 64, and 64 units, respectively, using the ReLU activation function. The output layer has a single unit with a linear activation function, as this is a regression problem. The model is then compiled with the mean squared error as the loss function and the Adam optimizer. Next, it is trained on the training data using a batch size of 32 for 120 epochs. To optimize the model, I normalized it by dividing every element of the data by the largest value in the data frame.

The NN model is created by predicting the GDP per capita value for the year 2021 and then evaluating it with the real value. The values of the loss and validation loss are equal to 1.8553e-05 and 1.8836e-05, respectively. These results are highly promising, indicating that the model has achieved very low error rates during training and validation. 

After the model is fully trained, it is used to predict the GDPs of every country in the future years. In my app, it predicts GDPs until the year 2040, but this can be easily changed. The model then creates and saves charts (GDP/years) for each country's predictions.

### Cloud Architecture

This project leverages AWS services for its infrastructure. AWS Amplify hosts the project's website, while AWS API Gateway is employed to create RESTful endpoints for communication between the frontend and backend. AWS Lambda function is responsible for fetching right GDP plots stored in Amazon S3 buckets. Permissions are configured to enable Lambda to access and retrieve these resources from S3.


<p align="center">
  <img src="GDPAPP.drawio.png" alt="GUI Screenshot" width="600" height="350">
</p>


### App Gui
The app's GUI prompts the user to input the name of a country in uppercase. Once the country name is entered, the app displays a GDP (year) chart for that country if it is found in the dataset. The area highlighted in red indicates that the values for those years are not actual data points but predictions made by the model.

<p align="center">
  App Gui video on YouTube
</p>
<div align="center">
  <a href="https://www.youtube.com/watch?v=L73-4yRNwlM">
    <img src="https://img.youtube.com/vi/L73-4yRNwlM/0.jpg" alt="Video" style="display:block; margin:auto;">
  </a>
</div>


In this way, the app allows users to interactively explore the GDP trends of different countries and compare the actual historical data with the model's predictions. This visual representation provides valuable insights into how well the model performs in forecasting GDP per capita for various countries over time.

## Data Analysis
Despite the low values of the loss and validation loss in the model, it cannot be considered a reliable indicator of GDP values for an extended period of time. The model efficiently predicts values for the year 2021 in the training phase because it had the opportunity to learn and adjust its parameters based on the available data. However, GDP depends on numerous unpredictable factors that this model does not take into account. Nevertheless, the GDP values for a few years ahead may be relatively close to those predicted by the model, as long as there are no significant economic changes in those years. More plots can be seen [here](gdp-charts).

<p align="center">
  <img src="gdp-charts/Poland_gdp_plot.png" alt="GUI Screenshot" width="600" height="350">
</p>

<p align="center">
  <img src="gdp-charts/Brazil_gdp_plot.png" alt="GUI Screenshot" width="600" height="350">
</p>

<p align="center">
  <img src="gdp-charts/Burundi_gdp_plot.png" alt="GUI Screenshot" width="600" height="350">
</p>

<p align="center">
  <img src="gdp-charts/Syria_gdp_plot.png" alt="GUI Screenshot" width="600" height="350">
</p>

## License
This project is licensed under the MIT License - [License](GUI/LICENSE.txt).

## Contact
Email: mikolajczachorowski260203@gmail.com
