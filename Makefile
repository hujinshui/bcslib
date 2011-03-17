# The make file for BCSLib

# compiler configuration

CC = g++
CFLAGS = -I. -Wall

all: temp1

temp1: bin/temp1

bin/temp1: bcslib/base/basic_defs.h temp/temp1.cpp
	$(CC) $(CFLAGS) temp/temp1.cpp -o bin/temp1