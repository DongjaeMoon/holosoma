"""Read Vulkan GPU UUIDs without starting a simulation or a CUDA workload."""
import ctypes as c
import json
import subprocess


def graphics_index(physical_gpu=3):
    import isaacgym
    from pathlib import Path
    class Application(c.Structure):
        _fields_=[('type',c.c_uint32),('next',c.c_void_p),('name',c.c_char_p),('version',c.c_uint32),
                  ('engine',c.c_char_p),('engine_version',c.c_uint32),('api',c.c_uint32)]
    class CreateInfo(c.Structure):
        _fields_=[('type',c.c_uint32),('next',c.c_void_p),('flags',c.c_uint32),('app',c.POINTER(Application)),
                  ('layer_count',c.c_uint32),('layers',c.c_void_p),('extension_count',c.c_uint32),('extensions',c.c_void_p)]
    class DeviceID(c.Structure):
        _fields_=[('type',c.c_uint32),('next',c.c_void_p),('uuid',c.c_ubyte*16),('driver',c.c_ubyte*16),
                  ('luid',c.c_ubyte*8),('node',c.c_uint32),('valid',c.c_uint32)]
    class Properties(c.Structure):
        _fields_=[('type',c.c_uint32),('next',c.c_void_p),('properties',c.c_ubyte*4096)]
    # IsaacGym bundles an old loader whose enumeration differs from the host's.
    library=Path(isaacgym.__file__).parent/'_bindings/linux-x86_64/libvulkan.so.1'
    vk=c.CDLL(str(library))
    app=Application(type=0,name=b'holosoma-gpu-mapping',api=(1<<22)|(1<<12))
    info=CreateInfo(type=1,app=c.pointer(app)); instance=c.c_void_p()
    code=vk.vkCreateInstance(c.byref(info),None,c.byref(instance))
    if code: raise RuntimeError(f'vkCreateInstance: {code}')
    try:
        count=c.c_uint32()
        vk.vkEnumeratePhysicalDevices(instance,c.byref(count),None)
        devices=(c.c_void_p*count.value)()
        vk.vkEnumeratePhysicalDevices(instance,c.byref(count),devices)
        mapping={}
        for index,device in enumerate(devices):
            identity=DeviceID(type=1000071004)
            properties=Properties(type=1000059001,next=c.cast(c.pointer(identity),c.c_void_p))
            vk.vkGetPhysicalDeviceProperties2(c.c_void_p(device),c.byref(properties))
            mapping[index]=bytes(identity.uuid).hex()
        uuid=subprocess.check_output(['nvidia-smi','-i',str(physical_gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        wanted=uuid.removeprefix('GPU-').replace('-','') if hasattr(uuid,'removeprefix') else uuid[4:].replace('-','')
        matches=[i for i,value in mapping.items() if value==wanted]
        if len(matches)!=1: raise RuntimeError(f'Cannot map GPU {physical_gpu}: {uuid}; Vulkan {mapping}')
        return dict(physical_gpu=physical_gpu,uuid=uuid,graphics_index=matches[0],vulkan_devices=mapping)
    finally:
        vk.vkDestroyInstance(instance,None)


if __name__=='__main__':
    print(json.dumps(graphics_index(),indent=2))
