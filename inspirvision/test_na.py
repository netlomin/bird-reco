import api
import os
import re


def fmt_bird_name(name):
    return re.sub(r'[^a-zA-Z]', ' ', name).lower()


if __name__ == '__main__':
    path = '../data_na'
    recos = []
    for root, dirs, files in os.walk(path):
        dirname = os.path.basename(root)

        for filename in files:
            img_file = path + '/' + dirname + '/' + filename
            bird_name = fmt_bird_name(dirname.split('.')[1])

            ret = api.reco_file(img_file)
            ret['file_name'] = filename
            ret['bird_name'] = bird_name
            ret['en_name'] = fmt_bird_name(ret['en_name'])
            recos.append(ret)

    s = 0
    v = 0
    i = 0
    for e in recos:
        msg = '' if 'message' not in e else ',错误:' + e.get('message')
        # 打印每个元素
        print(f'文件:{e["file_name"]},鸟名：{e["bird_name"]},识别鸟名:{e["en_name"]},置信度:{e["confidence"]}{msg}')

        if len(e['en_name']) >= 1:
            s += 1

        if e["confidence"] >= 0.9:
            v += 1
            if e['bird_name'] == e['en_name']:
                i += 1

    print('总数:{}，有效识别数：{}，识别正确率:{}'.format(len(recos), v, i))
    print('成功率:{}，有效率：{}，正确率:{}'.format(s / len(recos), v / len(recos), i / v))
