SRC = test.cu
EXE = ../build/test.exe
HEADER = matmul.cuh

FLAGS = -arch=sm_90 -lgtest -lgtest_main --cudart=shared

$(EXE): $(SRC) $(HEADER) 
	nvcc $(FLAGS) $(SRC) -o $(EXE)

