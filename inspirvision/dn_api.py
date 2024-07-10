import requests


def reco_file(path: str):
    with open(path, 'rb') as file:
        files = {'file': file}
        params = {"async": 0, "sc": 'web'}
        response = requests.post(url='http://www.dongniao.net/niaodian2', data=params, files=files)
        print(response.text)
        json = response.json()

    en_name = json[0][3]
    cn_name = json[0][2]
    confidence = json[0][0] / 100

    return {
        'en_name': en_name,
        'cn_name': cn_name,
        'confidence': confidence
    }


if __name__ == '__main__':
    # print(get_access_token())
    print(reco_file('../data_na/01.Ruby-throated hummingbird/01-Ruby-throated hummingbird.png'))
