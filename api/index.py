from http.server import HTTPServer, BaseHTTPRequestHandler
import multiprocessing
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import json
from flask import Flask, render_template, send_from_directory, request, jsonify
from music21 import *
data = {'result': 'error'}
host = ('', 80)
result = {
        'result': "error"
    }
import base64
import paddle
import paddle.nn as nn
import numpy as np
from Reader import Reader
import Seq2Seq
from flask import Flask, request
from Crypto.Cipher import AES
from flask_cors import *
from binascii import b2a_hex
app = Flask(__name__)

CORS(app, supports_credentials=True)
batch_size = 10
train_reader = Reader(batch_size, './work/data')
import json
import fractions
import time
# 初始化log写入器

# 模型参数设置
embedding_size = 256
hidden_size = 256
num_layers = 1

# 训练参数设置
epoch_num = 5000
learning_rate = 1e-5
log_iter = 200

# 定义一些所需变量
global_step = 0
log_step = 0
max_acc = 0

midi_model = Seq2Seq.Midi_Model(
    char_len=0x9FFF,  # 基本汉字的Unicode码范围为4E00-9FA5,这里设置0x9FFF长，基本够用
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_size=batch_size)
dur_model = Seq2Seq.Duration_Model(
    char_len=200,  # midi范围一般在100左右,这里设置200长，基本够用
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_size=batch_size)
midi_model.set_state_dict(paddle.load('Midi_Model/final_model'))
dur_model.set_state_dict(paddle.load('Duration_Model/final_model'))
def jiemihanshu(mima, readdata, iv):
    secret = mima  # 由用户输入的16位或24位或32位长的初始密码字符串
    iv = iv.encode('UTF-8')  # 随机获取16位变量
    encrypt_data = bytes().fromhex(readdata)
    cipher = AES.new(secret.encode('UTF-8'), AES.MODE_CBC, iv)
    decrypt_data = cipher.decrypt(encrypt_data)
    return decrypt_data
def valkey(time_en):
    if int(round(time.time() * 1000))-(int(jiemihanshu('20060815200608152006081520060815', bytes.decode(b2a_hex(base64.b64decode(time_en))),'2006081520060815').decode('UTF-8')[0:13]))<=30000:
        return True
    else:
        return False

@app.route('/api', methods=['GET'])
def test_get():
    # 解析请求参数
    param = request.args.to_dict()
    print(param)
    key = param['time']
    input = param['input']
    if valkey(key):
        if len(input) <= 100000:
            input_lyrics = input
            print(type(input))
            lyrics = []
            for i, lyric in enumerate(input_lyrics.replace('\n', '')):
                if i % batch_size == 0:
                    lyrics.append([])
                lyrics[i // batch_size].append(ord(lyric))
            while len(lyrics[-1]) % batch_size != 0:
                lyrics[-1].append(ord('#'))
            lyrics = paddle.to_tensor(lyrics)

            params_dict = paddle.load('Midi_Model/best_model')
            midi_model.set_dict(params_dict)

            # 设置为评估模式
            midi_model.eval()

            # 模型推理
            out = midi_model(lyrics)

            # 结果转换
            results = []
            for _ in np.argmax(out.numpy(), -1).reshape(-1):
                results.append(_)

            midis = []
            dur_dic = {}
            with open('dur_dic.json', 'r') as f:
                dur_str = f.readline()
                dur_dic = json.loads(dur_str)
            for i, midi in enumerate(results):
                if i % batch_size == 0:
                    midis.append([])
                midis[i // batch_size].append(midi) if midi <= 200 else midis[i // batch_size].append(0)
            while len(midis[-1]) % batch_size != 0:
                midis[-1].append(0)
            midis = paddle.to_tensor(midis)

            params_dict = paddle.load('Duration_Model/best_model')
            dur_model.set_dict(params_dict)

            # 设置为评估模式
            dur_model.eval()

            # 模型推理
            # out = nn.Softmax(dur_model(midis))
            out = dur_model(midis)

            # 结果转换
            durations = []
            for _ in np.argmax(out.numpy(), -1).reshape(-1):
                durations.append(_)

            dur_dic = {}
            with open('dur_dic.json', 'r') as f:
                dur_str = f.readline()
                dur_dic = json.loads(dur_str)
                print(dur_dic)

            stream1 = stream.Stream()
            for i, lyric in enumerate(input_lyrics.replace('\n', '')):
                if results[i] != 0:
                    n1 = note.Note(results[i])
                else:
                    n1 = note.Rest()
                n1.addLyric(lyric)
                n1.duration = duration.Duration(dur_dic[str(durations[i])])
                stream1.append(n1)
            import random
            name = ''
            for i in range(8):
                name += str(random.randint(0, 9))
            stream1.write("xml", './results/' + name + ".xml")
            stream1.write('midi', './results/' + name + '.midi')
            output = 'http://82.157.179.249:8080' + '/download/' + name + '.midi'
            result['result'] = output
        else:
            result['result'] = 'too lang'
    else:
        result['result'] = 'the key is wrong'
    # 返回json
    result_json = json.dumps(result)
    return result_json





@app.route("/download/<path:filename>")
def downloader(filename):
    dirpath = os.path.join(app.root_path+'/results')  # 这里是下在目录，从工程的根目录写起，比如你要下载static/js里面的js文件，这里就要写“static/js”
    return send_from_directory(dirpath, filename, as_attachment=True)  # as_attachment=True 一定要写，不然会变成打开，而不是下载

app.run(debug=True, host='0.0.0.0', port=8080)


