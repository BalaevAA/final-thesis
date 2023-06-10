## Выпускная квалификационная работа
### на тему: "Федеративное глубокое обучение на грид-системах"

CIFAR10/100
``` py
python main_fed.py --dataset cifar --model vgg16 --epochs 150 --gpu 0 --lr 0.01 --data_method --num_classes 10 --num_users 50 --data_dist noniid1
```

Tiny-ImageNet-200
```py
python main_fed.py --dataset imagenet --model mobilenet --epochs 100 --gpu 0 --lr 0.001 --data_method --num_classes 10 --num_users 10 --data_dist iid1
```
