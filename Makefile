CC = gcc
CFLAGS = -O1 -std=c99

all: 
	$(CC) $(CFLAGS) $(OBJS) kernel_perf.c -lm  -o kernel_perf.x -march=native
run:
	./kernel_perf.x