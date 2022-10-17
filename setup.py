from yadisk import YaDisk
import os

d = YaDisk()

try:
    os.mkdir('default_test_model')
except OSError as error:
    print(error)

os.system('cp default_test_config.json default_test_model/config.json')

print('downloading best model weights...')
d.download_public('https://disk.yandex.ru/d/nPvCYBUDX15G2A', 'default_test_model/checkpoint.pth')
print('download complete')
