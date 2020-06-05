from predictM.Predictor import Predictor


target_performance = [
    'carts-qps(2xx):0',
    'carts-qps(2xx):1',
    'carts-qps(2xx):2',
    'carts-qps(2xx):3',
    'catalogue-qps(2xx):0',
    'catalogue-qps(2xx):1',
    'catalogue-qps(2xx):2',
    'catalogue-qps(2xx):3',
    'front-end-latency:0',
    'front-end-latency:1',
    'front-end-latency:2',
    'front-end-latency:3',
    'front-end-latency:4',
    'front-end-latency:5',
    'front-end-latency:6',
    'front-end-latency:7',
    'front-end-qps(2xx):0',
    'front-end-qps(2xx):1',
    'front-end-qps(2xx):2',
    'front-end-qps(2xx):3',
    'orders-qps(2xx):0',
    'orders-qps(2xx):1',
    'orders-qps(2xx):2',
    'orders-qps(2xx):3',
    'payment-qps(2xx):0',
    'payment-qps(2xx):1',
    'payment-qps(2xx):2',
    'payment-qps(2xx):3',
    'payment-qps(2xx):4',
    'shipping-qps(2xx):0',
    'shipping-qps(2xx):1',
    'shipping-qps(2xx):2',
    'shipping-qps(2xx):3',
    'shipping-qps(2xx):4',
    'user-qps(2xx):0',
    'user-qps(2xx):1',
    'user-qps(2xx):2',
    'user-qps(2xx):3',
    'user-qps(2xx):4',
]

entity_list = []
with open('./benchmarks/FKB/entity2id.txt') as f:
    f.readline()
    for line in f.readlines():
        entity_list.append(line.rstrip().split('\t')[0])
f.close()

predict_result = {}
predictor = Predictor()
predictor.data_load()

for i in target_performance:
    if i in entity_list:
        result = predictor.predict(i)
        predict_result[i] = result
    else:
        predict_result[i] = 'invalid'
pass

with open('predict_result_m.txt', 'w') as f:
    f.write(str(predict_result))
f.close()