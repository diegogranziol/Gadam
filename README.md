# Code Submission for Iterate Averaging Helps: An Alternative Perspective in Deep Learning

## Environment
The code has been written in Python 3.7, Pytorch 1.1 (it should run in Pytorch 1.2, too. But there are unsuppressed, 
recurrent warnings if PyTorch 1.2 is used when running the RNN-based models). *The data-loader is of the torchvision-0.4.0 version*, and needs to be modified if a higher version torchvision is used.

## Packages
0. Python >= 3.6 + Anaconda
1. PyTorch >= 1.1.0
2. Torchvision <= 0.4.0
3. Tabulate
4. tqdm



## Run experiments on Vision tasks (CIFAR-10/100, ImageNet 32x32)
1. Download the CIFAR-10/100/ImageNet data into a folder. Assuming this is saved in data/ directory under the root path (Note: as per request from ImageNet, the downloading facility of the imagenet data loader in the code submission is removed.)
2. Run run_vision.py. Here we provide some examples

```bash
# Run SGD on VGG-16 (with batch normalisation) on CIFAR-100. This should give a test accuracy of around 74.2%
python3 -u run_vision.py --dir out/ --data_path data/ --dataset CIFAR100 --model VGG16BN --lr_init 0.1 --epochs 300 --use_test
# Run Gadam, with initial lr = 0.001 and IA lr = 0.0005 on CIFAR-100 in PreResNet-110 architecture. You should expect around 77.4%
python3 -u run_vision.py --dir out/ --data_path data/ --dataset CIFAR100 --model PreResNet110 --lr_init 0.001 --optim Gadam --ia_lr 0.0005 --wd 0.1 --use_test
# Run GadamX on ResNeXt-29. You should expect around 83.4% (ref: the baseline [1] is 82.2%)
python3 -u run_vision.py --dir out/ --data_path data/ --dataset CIFAR100 --model ResNeXt29CIFAR --lr_init 0.1 --optim GadamX --ia_lr 0.05 --wd 0.003 --use_test
# Some ImageNet experiment: run WideResNet28x10 with GadamX for 50 epochs. This should give you around 84.8% in Top 5 accuracy (ref: baseline in [2] is around 81%). --linear_annealing forces IA experiments to have the same learning rate schedule as the conventional ones.
python3 -u run_vision.py --dir out/ --data_path data/ --dataset ImageNet32 --model WideResNet28x10 --lr_init 0.03 --optim GadamX --wd 3e-4 --epochs 50 --linear_annealing --use_test
```

'3: (Experimental) To run with Lookahead optimiser (Gadam + LH), here we provide some examples:
```bash
# This runs the standard Lookahead with SGD as the base optimiser
python3 -u run_vision.py --dir out/ --data_path data/ --dataset CIFAR100 --model VGG16BN --lr_init 0.1 --epochs 300 --use_test --lookahead
# To run Gadam + LH on the same architecture:
python3 -u run_vision.py --dir out/ --data_path data/ --dataset CIFAR100 --model VGG16BN --optim Gadam --lr_init 0.0005 --epochs 300 --use_test --lookahead --ia_lr 0.00025 --ia_start 161
# This should reproduce a comparable result as in Appendix B 
```

## Run experiments on Language tasks (PTB) with LSTM
1. Run the scripts available at https://github.com/salesforce/awd-lstm-lm to download the PTB data into a directory as
you wish. Assuming this is again saved in data/, ...
2. Run run_language.py:

```bash
# Run baseline ASGD (featured in [3]). If you add --seed=141, this should give perplexity of around 61.2/58.8 (val/test)
python3 -u run_language.py --data_path data/ --dir out/ --dataset PTB --epochs 500 --save PTB.pt

# To achieve the similar baseline we obtained in 200 epochs, simply replace --epochs 500 with --epochs 200
python3 -u run_language.py --data_path data/ --dir out/ --dataset PTB --epochs 200 --save PTB.pt

# To run Gadam, 
python3 -u run_language.py --data_path data/ --dir out/ --dataset PTB --epochs 200 --save PTB.pt --optimizer Gadam --ia_start 100 --lr 0.03
# This should give perplexity of around 61.4/58.7 (val/test) in 200 epochs
```

## Acknowledgements:
We use and thank materials from the following individuals/repositories:
1. https://github.com/salesforce/awd-lstm-lm [for Language models]
2. https://github.com/timgaripov/swa/ [for SWA implementation]
3. https://github.com/uclaml/Padam [for Padam]
4. https://github.com/michaelrzhang/lookahead [for Lookahead]
5. https://github.com/bearpaw/pytorch-classification [for PreResNet]
6. Pau Rodríguez López (pau.rodri1@gmail.com) [for ResNeXt]
7. https://github.com/meliketoy/wide-resnet.pytorch [for WideResNet]

## References:
[1] Xie, S., Girshick, R., Dollár, P., Tu, Z. and He, K., 2017. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1492-1500).

[2] Chrabaszcz, P., Loshchilov, I. and Hutter, F., 2017. A downsampled variant of ImageNet as an alternative to the CIFAR datasets. arXiv preprint arXiv:1707.08819.

[3] Merity, S., Keskar, N.S. and Socher, R., 2017. Regularizing and optimizing LSTM language models. arXiv preprint arXiv:1708.02182.