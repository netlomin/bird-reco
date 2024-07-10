import boto3
from botocore.exceptions import ClientError
import time

# 配置AWS的访问密钥和秘密密钥
accessKeyId = 'ASIA47CRVV3OUTZR2GPN'
secretAccessKey = 'WhOIks+ku75bx8RS1QlBxtuuYQM5zK0kD2e0/mO4'
session_token = 'IQoJb3JpZ2luX2VjEHcaCXVzLWVhc3QtMiJIMEYCIQDIFeDILbg3kkcOgF95l34tLKvuLjpOeyOGuG1pGfiNQAIhANadZaarl2BnNzHuEwIUZME3MO+dAHO/oZc9byXItT2oKooECEAQABoMODkxMzc3MDA0MjUzIgxHGdmsLToxGXdPHJIq5wMhw4AxRC/4AmY3VPN5T+UvJ2dlR2LT2iFrkLUTX3d3tnpHnYkRX38M2anfHFGCcIvOA9zymWA/REH3DlsmkqfaFtW7mQtU3JvXJoRMGC/hQLWQNW8Qa0Ef7tg9lGj6JHUIph3fiYJygShO59kdbXCzu2Y50UlCTREV3t4gWnwPyuA2R+sYbL0azwTjvk72mGEf0T7YAzdXDAOkqxsdGDb6AabWW3wkw2UShVVK0rDeWJ5TVKiLRWzmptc88V0rmHUNApfEHbA/P6Vcjh4Ef0UkmuCSUlCFCA8NRRZgI534SVLGpacbvLVxxZ4WjTVCP0F8MeofYJs4ha3pCn+9AMJC+jcZZ6Akfu3ji5c/ASBgxz7i8YEzH07TcBSnozzEHd0a6gEM8pJgkpJgmC1g5iD42LFZ0tJ0mb86VAvhyn9eFbZ2J3vYllbzInJmTA0Ks5mVnh1lqywVx+wnf3qMFecgN991NTK/aczYPOmkNaOwPtAStLz1OCKXagx66nz2fHTdxiGHSLlylclbdo8HUy15bPIE7bh3mpIZIon8Sn8No75aXz+7GsD2DPhcwJKYkMIgbC6dN4jUQH7tduam86DZk585G7qqp6YaohG/PQwXkSESz2M8yjK0oaz9pFgYuM6CFBnc2ty9MLzquLQGOpwBsrzmkRS6GR0viVPluUzh+OcXv4ePvQMs05E/g00J6sLWDpwCKlFw27u7YMeu6UcUP2x1+Q7yH3ZZwNg/ogjhyo+n/AVF3YkdvJTPQgBVZPWpVNs46NO8JRs6iVT6DTw0c9VUPXOom6/K++MHkCek+c3PYJ6mrhzq0mR5S9nGpLOWhOtWndqI+PMCJRHs2mhnHJES2JJuBU9jttIg'

# 初始化boto3客户端
s3_client = boto3.client('s3', aws_access_key_id=accessKeyId, aws_secret_access_key=secretAccessKey,
                         aws_session_token=session_token, region_name='us-east-2')


def upload(file_name, bucket_name, key):
    # 使用upload_file方法上传文件
    try:
        s3_client.upload_file(file_name, bucket_name, key)
        print(f"File {file_name} uploaded successfully to S3 bucket {bucket_name}")
    except ClientError as e:
        print(f"Error uploading file: {e}")


if __name__ == '__main__':
    file_path = 'ts-event-101454-101458-4400.ts'
    key = 'record/us1791280608554299392/rn0142c293Ec054B/20240711/1/test-'

    for i in range(0, 50):
        ts = int(time.time())
        upload(file_path, 'ipc-cloud-prod-891377004253', key + str(i))
        print('time=', int(time.time()) - ts)
