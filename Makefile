CC = gcc
CFLAGS = -O1 -std=c99

all: 
	$(CC) $(CFLAGS) $(OBJS) bilinear.c -lm  -o bilinear.x -march=native
run:
	./bilinear.x

clean:
	rm -f *.x *~ *.o
