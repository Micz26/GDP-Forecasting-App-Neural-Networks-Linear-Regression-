import os
import boto3


aws_access_key_id = 'access-key'
aws_secret_access_key = 'secret-access-key'
bucket_name = 'gdp-charts'

charts_directory = 'C:\\Users\\mikol\\OneDrive\\Pulpit\\GDP Forecasting\\gdp-charts'

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

for file_name in os.listdir(charts_directory):
    if file_name.endswith('.png'):
        file_path = os.path.join(charts_directory, file_name)
        s3.upload_file(file_path, bucket_name, file_name)
