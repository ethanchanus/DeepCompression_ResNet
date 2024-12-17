
python3 train.py --dataset=cifar-10 --epochs=100 --model=resnet-50 
python3 weight_quantization.py  --dataset=cifar-10 saves_resnet-50_cifar-10/model_after_retraining.ptmodel

python3 huffman_encode.py --dataset=cifar-10  --model=resnet-50 saves_resnet-50_cifar-10/model_after_weight_sharing.ptmode

#python3 train.py --dataset=cifar-10 --epochs=160 --model=resnet-50  --resumetrain=results/saves_resnet-50_cifar-10/initial_model.ptmodel.150  --ignoretrain=1 --epochretrain=100

#python3 weight_quantization.py  --dataset=cifar-10 results/saves_resnet-50_cifar-10\model_after_retraining.ptmodel.100

#python3 huffman_encode.py --dataset=cifar-10  --model=resnet-50 results/saves_resnet-50_cifar-10/model_after_weight_sharing.ptmodel