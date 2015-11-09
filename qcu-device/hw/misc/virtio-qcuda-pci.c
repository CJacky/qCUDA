#include "hw/pci/pci.h"
#include "hw/virtio/virtio.h"
#include "hw/virtio/virtio-bus.h"
#include "hw/virtio/virtio-pci.h"
#include "hw/virtio/virtio-qcuda.h"

static Property virtio_qcuda_pci_properties[] = {
    DEFINE_PROP_BIT("ioeventfd", VirtIOPCIProxy, flags,
                    VIRTIO_PCI_FLAG_USE_IOEVENTFD_BIT, true),
    DEFINE_PROP_UINT32("vectors", VirtIOPCIProxy, nvectors, 2),
    DEFINE_PROP_END_OF_LIST(),
};

static void virtio_qcuda_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    VirtIOQCPCI *qcu= VIRTIO_QC_PCI(vpci_dev);
    DeviceState *vdev = DEVICE(&qcu->vdev);

    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));
    object_property_set_bool(OBJECT(vdev), true, "realized", errp);
}

static void virtio_qcuda_pci_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    dc->props = virtio_qcuda_pci_properties;

    k->realize = virtio_qcuda_pci_realize;
    pcidev_k->vendor_id = PCI_VENDOR_ID_REDHAT_QUMRANET;
    pcidev_k->device_id = PCI_DEVICE_ID_VIRTIO_QC;
	pcidev_k->revision = VIRTIO_PCI_ABI_VERSION;
   	pcidev_k->class_id  = PCI_CLASS_OTHERS;
}

static void virtio_qcuda_pci_instance_init(Object *obj)
{
    VirtIOQCPCI *dev = VIRTIO_QC_PCI(obj);

    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),
                                TYPE_VIRTIO_QC);
}

static const TypeInfo virtio_qcuda_pci_info = {
    .name          = TYPE_VIRTIO_QC_PCI,
    .parent        = TYPE_VIRTIO_PCI,
    .instance_size = sizeof(VirtIOQCPCI),
    .instance_init = virtio_qcuda_pci_instance_init,
    .class_init    = virtio_qcuda_pci_class_init,
};

static void virtio_qcuda_pci_register_types(void)
{
    type_register_static(&virtio_qcuda_pci_info);
}
type_init(virtio_qcuda_pci_register_types)
