/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow-lite/core/device.h"

static const size_t OFLITE_DEVICE_COUNT_LIMIT = 64;
static const size_t OFLITE_DEVICE_TYPE_LENGTH_LIMIT = 128;

typedef struct OfLiteDeviceFactoryItem {
  char type[OFLITE_DEVICE_TYPE_LENGTH_LIMIT];
  OfLiteDeviceFactory factory;
} OfLiteDeviceFactoryItem;

typedef struct OfLiteDeviceRegistry {
  size_t size;
  OfLiteDeviceFactoryItem items[OFLITE_DEVICE_COUNT_LIMIT];
} OfLiteDeviceRegistry;

OfLiteDeviceRegistry* GetOfLiteDeviceRegistry() {
  static OfLiteDeviceRegistry oflite_device_registry{.size = 0};
  return &oflite_device_registry;
}

OFLITE_API void OfLiteDeviceCreate(OfLiteStringRef type, size_t ordinal,
                                   OfLiteDevice** device) {
  OfLiteDeviceRegistry* registry = GetOfLiteDeviceRegistry();
  for (size_t i = 0; i < registry->size; ++i) {
    const OfLiteDeviceFactoryItem& item = registry->items[i];
    if (0 == strcmp(type.data, item.type)) {
      *device = item.factory(ordinal);
      return;
    }
  }
  // TODO(): create device error
}

OFLITE_API void OfLiteDeviceDestory(OfLiteDevice* device) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->destory(device);
}

OFLITE_API void OfLiteDeviceQueryId(const OfLiteDevice* device,
                                    OfLiteDeviceId* id) {
  reinterpret_cast<const OfLiteDeviceVTable*>(device)->query_id(device, id);
}

OFLITE_API void OfLiteDeviceQueryName(const OfLiteDevice* device,
                                      OfLiteStringRef* name) {
  reinterpret_cast<const OfLiteDeviceVTable*>(device)->query_name(device, name);
}

OFLITE_API void OfLiteDeviceQueryOrdinal(const OfLiteDevice* device,
                                         size_t* ordinal) {
  reinterpret_cast<const OfLiteDeviceVTable*>(device)->query_ordinal(device,
                                                                     ordinal);
}

OFLITE_API void OfLiteDeviceCreateEvent(OfLiteDevice* device,
                                        OfLiteEvent** event) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->create_event(device, event);
}

OFLITE_API void OfLiteDeviceCreateStream(OfLiteDevice* device,
                                         OfLiteStream** stream) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->create_stream(device, stream);
}

OFLITE_API void OfLiteDeviceMalloc(OfLiteDevice* device, size_t size,
                                   void** ptr) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->malloc(device, size, ptr);
}

OFLITE_API void OfLiteDeviceFree(OfLiteDevice* device, void* ptr) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->free(device, ptr);
}

OFLITE_API void OfLiteDeviceMallocHost(OfLiteDevice* device, size_t size,
                                       void** ptr) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->malloc_host(device, size, ptr);
}

OFLITE_API void OfLiteDeviceFreeHost(OfLiteDevice* device, void* ptr) {
  reinterpret_cast<OfLiteDeviceVTable*>(device)->free_host(device, ptr);
}

OFLITE_API void OfLiteDeviceRegisterFactory(OfLiteStringRef type,
                                            OfLiteDeviceFactory factory) {
  if (type.size >= OFLITE_DEVICE_TYPE_LENGTH_LIMIT) {
    // TODO(): the length of device type name should less than 128
    return;
  }
  OfLiteDeviceRegistry* registry = GetOfLiteDeviceRegistry();
  if (registry->size == OFLITE_DEVICE_COUNT_LIMIT) {
    // TODO(): failed to register device
    return;
  }
  OfLiteDeviceFactoryItem* item = &registry->items[registry->size];
  strncpy(item->type, type.data, OFLITE_DEVICE_TYPE_LENGTH_LIMIT - 1);
  item->type[type.size] = 0;
  item->factory = factory;
  registry->size += 1;
}
