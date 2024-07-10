import time
import hmac
import hashlib

if __name__ == '__main__':
    uuid = 'rn01Ce054ab8e57E'
    secret = '065e4e23c99148139643a14813891adc'
    print('client-id:', 'rlink_' + uuid + '_V2')

    ts = int(time.time())
    print('username:', uuid + '|' + 'signMethod=hmacSha256,ts=' + str(ts))

    content = 'uuid=' + uuid + ',ts=' + str(ts)
    content_bytes = content.encode('utf-8')
    secret_bytes = secret.encode('utf-8')
    print('password:', hmac.new(secret_bytes, content_bytes, digestmod=hashlib.sha256).hexdigest())
