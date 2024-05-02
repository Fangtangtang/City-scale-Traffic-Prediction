import csv
import jsonlines

keys=['\ufeffid', 'start_date', 'end_date', 'label', 'downstream_id', 'downstream_label', 'upstream_id', 'upstream_label', 'geo_point_2d', 'geo_shape']

# 打开CSV文件
with open('data\\raw\\geo_reference.csv', newline='',encoding='utf-8') as csvfile:
    # 创建一个CSV读取器
    csv_reader = csv.DictReader(csvfile, delimiter=';')
    print(csv_reader.fieldnames)
    for row in csv_reader:
        dict={}
        for key in keys:
            if key==keys[0]:
                dict["id"]=row[key]
            else:
                dict[key]=row[key]
    
        with jsonlines.open("data\\pre_processed\\geo_reference.jsonl","a") as file:
            file.write(dict)

