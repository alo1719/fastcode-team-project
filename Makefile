compile:
	g++ -I/usr/lib64/ -lopencv_core -lopencv_highgui -fopenmp -mavx -mavx2 -mfma -O3 test_bilinear.cpp -o test_bilinear.x

run:
	./test_bilinear.x 800 1600

clean:
	rm -rf *.x