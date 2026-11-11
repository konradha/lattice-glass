.PHONY: all cpu metal cuda cuda-depth cuda-throughput cuda-ampere help
all: cpu
cpu:
	$(MAKE) -C cpu
metal:
	$(MAKE) -C metal
cuda:
	$(MAKE) -C cuda
cuda-depth:
	$(MAKE) -C cuda driver-depth
cuda-throughput:
	$(MAKE) -C cuda driver-throughput
cuda-ampere:
	$(MAKE) -C cuda driver-ampere
help:
	@printf '%s\n' 'make cpu | metal | cuda-depth | cuda-throughput | cuda-ampere' 'python3 driver.py --help' 'No training, simulations, or benchmarks run by build targets.'
