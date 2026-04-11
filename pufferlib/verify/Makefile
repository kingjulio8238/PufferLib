CC ?= cc
CFLAGS = -O2 -Wall -Wextra -fPIC
LDFLAGS = -lm

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LIB = libpuffer_verify.dylib
	SHARED_FLAGS = -dynamiclib
else
	LIB = libpuffer_verify.so
	SHARED_FLAGS = -shared
endif

# OpenMP support (optional)
OMP_FLAGS :=
ifneq ($(NO_OMP), 1)
	ifeq ($(UNAME), Darwin)
		# macOS: check for libomp from homebrew
		ifneq ($(wildcard /opt/homebrew/opt/libomp/lib/libomp.dylib),)
			OMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
		endif
	else
		OMP_FLAGS = -fopenmp
	endif
endif

.PHONY: all clean test

all: $(LIB)

$(LIB): math_verify.c math_verify.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $(OMP_FLAGS) -o $@ math_verify.c $(LDFLAGS)

test_verify: test_verify.c math_verify.c math_verify.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ test_verify.c math_verify.c $(LDFLAGS)

test: test_verify
	./test_verify

clean:
	rm -f $(LIB) libpuffer_verify.dylib libpuffer_verify.so test_verify
