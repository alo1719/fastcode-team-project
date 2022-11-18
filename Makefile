CC = gcc
CFLAGS = -O1 -std=c99

all: 
	$(CC) $(CFLAGS) $(OBJS) kernel_perf.c -lm  -o kernel_perf.x -march=native
run:
	./kernel_perf.x

clean:
	rm -f *.x *~ *.o

assemble:
	objdump -s -d -f --source ./kernel_perf.x > kernel_perf.S