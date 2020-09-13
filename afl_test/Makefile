
all:
	~/AFLplusplus/afl-clang-fast -O2 ./foo.c

fuzz:
	rm -f shared_memory.shm
	AFL_SKIP_CPUFREQ=1 ~/AFLplusplus/afl-fuzz -d -i inputs -o outputs ./a.out 1000000000 @@

internal: all
	./a.out internal

