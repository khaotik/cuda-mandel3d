rm -f a.out a.png a.raw
nvcc -gencode arch=compute_86,code=sm_86 a.cu && ./a.out && python3 imshow.py
