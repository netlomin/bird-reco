import requests
import base64

access_key = 'APPID_TL6im79QUD9b6Q28'
access_secret = '640247f9264e0e99d8247d0a5e61ca88'
access_token = None


def get_access_token() -> str:
    url = f'https://ai.inspirvision.cn/s/api/getAccessToken?accessKey={access_key}&accessSecret={access_secret}'
    response = requests.get(url)
    json = response.json()
    if json['status'] == 200:
        return json['data']['access_token']
    else:
        raise ValueError(json['message'])


def reco(img_base64: str):
    global access_token
    if access_token is None:
        access_token = get_access_token()

    params = {"token": access_token, "imgBase64": img_base64}
    response = requests.post(url='https://ai.inspirvision.cn/s/api/birdType', data=params)
    json = response.json()
    if json['status'] == 200:
        if len(json['data']['pet']) <= 0 or len(json['data']['pet'][0]['identification']) <= 0:
            return {
                'pet_type': '',
                'en_name': '',
                'cn_name': '',
                'confidence': 0
            }

        pet_type = json['data']['petType']
        identification = json['data']['pet'][0]['identification'][0]
        en_name = identification['english_name']
        cn_name = identification['chinese_name']
        confidence = identification['confidence']
        return {
            'pet_type': pet_type,
            'en_name': en_name,
            'cn_name': cn_name,
            'confidence': confidence
        }
    else:
        return {
            'pet_type': '',
            'en_name': '',
            'cn_name': '',
            'confidence': 0,
            'message': json['message']
        }


def reco_file(path: str):
    with open(path, "rb") as file:
        file_content = file.read()
    base64_data = base64.b64encode(file_content)
    return reco(base64_data.decode('utf-8'))


if __name__ == '__main__':
    # print(get_access_token())
    print(reco_file('../birds/testing/canglu/1 (2).jpg'))
