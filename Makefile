PLATFORM=-mcpu=cortex-a8 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a8
OPTIMIZE=-O2 -ffast-math
WARN=-Wall -Wextra
CFLAGS=-std=gnu99 $(OPTIMIZE) $(PLATFORM) $(WARN) -fPIC
LDFLAGS=-fPIC
LDLIBS=-lrt -lpruio 

PYTHON_SO_NAME=neurobot_cffi.cpython-37m-arm-linux-gnueabihf.so

.PHONY : all
all : $(PYTHON_SO_NAME)

$(PYTHON_SO_NAME) : libneurobot.a build_neurobot.py
	python3 build_neurobot.py

libneurobot.a : libneurobot.o
	ar rcs $@ $<
