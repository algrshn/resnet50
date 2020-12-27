import configparser
config = configparser.ConfigParser()
config.read('config.txt')
buffer_folder=config.get('train','buffer_size')

print(buffer_folder)