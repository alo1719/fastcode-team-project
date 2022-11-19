CC = gcc
CFLAGS = -O3 -std=c99

all: 
	$(CC) $(CFLAGS) $(OBJS) kernel_perf.c -lm  -o kernel_perf.x -march=native
run:
	./kernel_perf.x
assemble:
	objdump -s -d -f --source ./kernel_perf.x > kernel_perf.S

clean:
	rm -rf *.x *.S