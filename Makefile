compile:
	g++ -I/usr/lib64/ -lopencv_core -lopencv_highgui -fopenmp -mavx -mavx2 -mfma -O3 testc.cpp -o testc.x

run:
	./testc.x 160 320

clean:
	rm -rf *.x
