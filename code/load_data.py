# -*- coding: UTF-8 -*-
import json
import codecs
import csv
import pprint


def read_json_text():
    """读取dataset.json里面的数据，将text放入文件中
    路径：data/temp/text.txt
    格式：i+1行（id） i+2行（user） i+3行（context） i+4行（labels）
    """
    txtfile = codecs.open('data/temp/text.txt', 'w', encoding='utf-8', errors='ignore')

    with open('Frames_dataset/frames.json', 'r') as file_object:
        contents = json.load(file_object)
        for i in range(len(contents)):
            for item in contents[i]['turns']:
                print(i, '****************************')
                print(item['author']+':   '+item['text'])
                for act in item['labels']['acts']:
                    print(act)
                print(item['labels']['active_frame'])
                try:
                    txtfile.write(str(i)+'\n')
                    txtfile.write(item['author']+'\n')
                    txtfile.write(item['text']+'\n')
                    txtfile.write('>>>>'+'\n')
                except:
                    txtfile.write('ERROR in {}'.format(i) + '\n')
        txtfile.close()


def write_csv():
    """将信息写到csv文件中"""
    csvfile = open('Frames_dataset/frames_label.csv', 'a+')
    writer = csv.DictWriter(csvfile, fieldnames=['id', 'user', 'text', 'label'])
    writer.writeheader()
    i = 0
    with open('Frames_dataset/frames.json', 'r') as file_object:
        contents = json.load(file_object)
    for item in contents[i]['turns']:
        print(item['author'] + ':   ' + item['text'])
        for act in item['labels']['acts']:
            print(act)
        label = input('>>>>>>>>')
        writer.writerow({'id': i, 'user': item['author'], 'text': item['text'], 'label': label})


def count_info():
    """统计label的个数"""
    txtfile = codecs.open('Frames_dataset/test.txt', 'r', encoding='utf-8', errors='ignore').readlines()
    label_list = []
    n_list = []
    for i in range(0, len(txtfile), 4):
        # print txtfile[i]
        # print txtfile[i+1]
        print(txtfile[i + 3])
        ll = txtfile[i + 3].replace('>>>> ', '').strip('\n').strip('\r').split(' ')
        label_list.extend(ll)
        n_list.append(len(ll))
    for t in set(label_list):
        print(t, label_list.count(t))
    for t in set(n_list):
        print(t, n_list.count(t))


def show_info():
    """输入id,显示相关的信息"""
    with open('../data/Frames_dataset/frames.json', 'r') as file_object:
        contents = json.load(file_object)
    while True:
        i = int(input('>>>'))
        for item in contents[i]['turns']:
            print(item['author']+':   '+item['text'])
            for act in item['labels']['acts']:
                print(act)
            print(item['labels']['active_frame'])


if __name__ == '__main__':
    show_info()
