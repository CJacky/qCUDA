# qCUDA

In start qemu vm's command, add "device virtio-hm-pci".


qemu-virthm:
	Hypervisior with virtio device. Using in host.

vhm-driver:
	virtio device driver. Using in guest.

vhm-library:
	cuda API. Using in guest.
	Note: before using this library, you should add "--enable-cuda" when compiler hypervisior. At Host side, it is using nvidia official driver.

vhm-test:
	sample code for write, read and cuda.
	
# qCUDA
# qCUDA
