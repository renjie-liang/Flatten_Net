CUDA_VISIBLE_DEVICES='0' python3 -u main.py  \
	--cfg configs/Flatten_CNN.yaml  \
	--bs 10  \
	--nw 4  \
	--name test_flatten_cnn \
	--debug
