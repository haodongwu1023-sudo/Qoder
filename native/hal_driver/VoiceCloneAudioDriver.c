/*
 * VoiceCloneAudioDriver.c
 * Minimal macOS CoreAudio HAL AudioServerPlugIn exposing two virtual devices:
 *   1. "VoiceClone Mic" (Virtual Input Microphone backed by mic.ring)
 *   2. "VoiceClone Meeting Audio" (Virtual Output Speaker backed by meeting.ring)
 */

#include <CoreAudio/AudioServerPlugIn.h>
#include <CoreAudio/AudioHardwareBase.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <os/log.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define VC_RING_DIR "/Users/Shared/VoiceCloneLive/Audio"
#define VC_MIC_RING_PATH "/Users/Shared/VoiceCloneLive/Audio/mic.ring"
#define VC_MEETING_RING_PATH "/Users/Shared/VoiceCloneLive/Audio/meeting.ring"
#define VC_RING_MAGIC 0x56435242U /* 'VCRB' */
#define VC_RING_CAPACITY_FRAMES (48000U * 4U) /* 4 seconds @ 48kHz */
#define VC_ZERO_TS_PERIOD 16384U /* Must be >= 10923 per AudioServerPlugIn.h */

#define kObjectID_PlugIn        kAudioObjectPlugInObject /* 1 */
#define kObjectID_MicDevice     2
#define kObjectID_MicStream     3
#define kObjectID_MeetingDevice 4
#define kObjectID_MeetingStream 5

#define kPlugInFactoryUUID CFSTR("8D4E2B61-7A9C-4E12-9F31-2026A8B40001")
#define kPlugInBundleID    CFSTR("com.voiceclone.audio.driver")

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t capacity_frames;
    _Atomic uint32_t reserved;
    _Atomic uint64_t write_pos;
    _Atomic uint64_t read_pos;
    float samples[VC_RING_CAPACITY_FRAMES * 2];
} VCRingBuffer;

typedef struct {
    const char *path;
    VCRingBuffer *ring;
    int fd;
    pthread_mutex_t lock;
} VCRingHandle;

typedef struct {
    AudioObjectID deviceID;
    AudioObjectID streamID;
    int isInput;
    CFStringRef name;
    CFStringRef uid;
    _Atomic uint32_t ioCount;
    Float64 sampleRate;
    Float64 hostTicksPerFrame;
    UInt32 bufferFrameSize;
    UInt64 numberTimeStamps;
    UInt64 anchorHostTime;
    VCRingHandle *ringHandle;
} VCDeviceState;

static pthread_mutex_t gPlugInLock = PTHREAD_MUTEX_INITIALIZER;
static UInt32 gRefCount = 0;
static AudioServerPlugInHostRef gHost = NULL;

static VCRingHandle gMicRing = { .path = VC_MIC_RING_PATH, .ring = NULL, .fd = -1, .lock = PTHREAD_MUTEX_INITIALIZER };
static VCRingHandle gMeetingRing = { .path = VC_MEETING_RING_PATH, .ring = NULL, .fd = -1, .lock = PTHREAD_MUTEX_INITIALIZER };

static VCDeviceState gDevices[2] = {
    {
        .deviceID = kObjectID_MicDevice,
        .streamID = kObjectID_MicStream,
        .isInput = 1,
        .name = CFSTR("VoiceClone Mic"),
        .uid = CFSTR("com.voiceclone.driver.mic"),
        .ioCount = 0,
        .sampleRate = 48000.0,
        .hostTicksPerFrame = 0.0,
        .bufferFrameSize = 512,
        .numberTimeStamps = 0,
        .anchorHostTime = 0,
        .ringHandle = &gMicRing,
    },
    {
        .deviceID = kObjectID_MeetingDevice,
        .streamID = kObjectID_MeetingStream,
        .isInput = 0,
        .name = CFSTR("VoiceClone Meeting Audio"),
        .uid = CFSTR("com.voiceclone.driver.meeting"),
        .ioCount = 0,
        .sampleRate = 48000.0,
        .hostTicksPerFrame = 0.0,
        .bufferFrameSize = 512,
        .numberTimeStamps = 0,
        .anchorHostTime = 0,
        .ringHandle = &gMeetingRing,
    }
};

static void vc_log_unknown(const char *fn, AudioObjectID objID, const AudioObjectPropertyAddress *addr) {
    UInt32 s = addr ? addr->mSelector : 0;
    UInt32 sc = addr ? addr->mScope : 0;
    UInt32 el = addr ? addr->mElement : 0;
    os_log_error(
        OS_LOG_DEFAULT,
        "VoiceCloneHAL %{public}s unknown prop obj=%u sel=%c%c%c%c(0x%x) scope=%c%c%c%c elem=%u",
        fn,
        (unsigned)objID,
        (int)((s >> 24) & 0xff), (int)((s >> 16) & 0xff), (int)((s >> 8) & 0xff), (int)(s & 0xff), (unsigned)s,
        (int)((sc >> 24) & 0xff), (int)((sc >> 16) & 0xff), (int)((sc >> 8) & 0xff), (int)(sc & 0xff),
        (unsigned)el
    );
}

static void vc_ensure_ring_dir(void) {
    mkdir("/Users/Shared/VoiceCloneLive", 0777);
    chmod("/Users/Shared/VoiceCloneLive", 0777);
    mkdir(VC_RING_DIR, 0777);
    chmod(VC_RING_DIR, 0777);
}

static void vc_open_ring(VCRingHandle *h, uint32_t sampleRate) {
    pthread_mutex_lock(&h->lock);
    if (h->ring != NULL) {
        h->ring->sample_rate = sampleRate;
        pthread_mutex_unlock(&h->lock);
        return;
    }
    vc_ensure_ring_dir();
    int fd = open(h->path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        os_log_error(OS_LOG_DEFAULT, "VoiceCloneHAL open %{public}s failed errno=%d", h->path, errno);
        pthread_mutex_unlock(&h->lock);
        return;
    }
    fchmod(fd, 0666);
    size_t mapSize = sizeof(VCRingBuffer);
    if (ftruncate(fd, (off_t)mapSize) != 0) {
        os_log_error(OS_LOG_DEFAULT, "VoiceCloneHAL ftruncate %{public}s failed errno=%d", h->path, errno);
        close(fd);
        pthread_mutex_unlock(&h->lock);
        return;
    }
    void *ptr = mmap(NULL, mapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        os_log_error(OS_LOG_DEFAULT, "VoiceCloneHAL mmap %{public}s failed errno=%d", h->path, errno);
        close(fd);
        pthread_mutex_unlock(&h->lock);
        return;
    }
    VCRingBuffer *rb = (VCRingBuffer *)ptr;
    if (rb->magic != VC_RING_MAGIC || rb->capacity_frames != VC_RING_CAPACITY_FRAMES) {
        memset(rb, 0, sizeof(VCRingBuffer));
        rb->magic = VC_RING_MAGIC;
        rb->version = 1;
        rb->channels = 2;
        rb->capacity_frames = VC_RING_CAPACITY_FRAMES;
        atomic_store(&rb->write_pos, 0);
        atomic_store(&rb->read_pos, 0);
    }
    rb->sample_rate = sampleRate;
    h->fd = fd;
    h->ring = rb;
    pthread_mutex_unlock(&h->lock);
}

static void vc_ring_read_stereo(VCRingHandle *h, float *out, UInt32 frames) {
    VCRingBuffer *rb = h->ring;
    if (!rb) {
        memset(out, 0, frames * 2 * sizeof(float));
        return;
    }
    uint64_t r = atomic_load_explicit(&rb->read_pos, memory_order_relaxed);
    uint64_t w = atomic_load_explicit(&rb->write_pos, memory_order_acquire);
    if (r > w) {
        r = w;
    }
    uint64_t avail = w - r;
    if (avail > rb->capacity_frames) {
        /* Overrun recovery: keep a small live window rather than rewinding the full 4s ring. */
        uint64_t keep = (rb->capacity_frames > frames) ? frames : rb->capacity_frames;
        r = w - keep;
        avail = keep;
    }
    UInt32 toRead = (avail >= frames) ? frames : (UInt32)avail;
    for (UInt32 i = 0; i < toRead; ++i) {
        uint64_t idx = ((r + i) % rb->capacity_frames) * 2;
        out[i * 2] = rb->samples[idx];
        out[i * 2 + 1] = rb->samples[idx + 1];
    }
    if (toRead < frames) {
        memset(out + toRead * 2, 0, (frames - toRead) * 2 * sizeof(float));
    }
    atomic_store_explicit(&rb->read_pos, r + toRead, memory_order_release);
}

static void vc_ring_write_stereo(VCRingHandle *h, const float *in, UInt32 frames) {
    VCRingBuffer *rb = h->ring;
    if (!rb) return;
    uint64_t w = atomic_load_explicit(&rb->write_pos, memory_order_relaxed);
    for (UInt32 i = 0; i < frames; ++i) {
        uint64_t idx = ((w + i) % rb->capacity_frames) * 2;
        rb->samples[idx] = in[i * 2];
        rb->samples[idx + 1] = in[i * 2 + 1];
    }
    atomic_store_explicit(&rb->write_pos, w + frames, memory_order_release);
}

static VCDeviceState *vc_find_device(AudioObjectID objID) {
    for (size_t i = 0; i < 2; ++i) {
        if (gDevices[i].deviceID == objID || gDevices[i].streamID == objID) {
            return &gDevices[i];
        }
    }
    return NULL;
}

static void vc_update_host_ticks(VCDeviceState *dev) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    Float64 ticksPerSec = ((Float64)info.denom / (Float64)info.numer) * 1000000000.0;
    dev->hostTicksPerFrame = (dev->sampleRate > 0.0) ? (ticksPerSec / dev->sampleRate) : 0.0;
}

static AudioStreamBasicDescription vc_make_asbd(Float64 sr) {
    AudioStreamBasicDescription asbd;
    memset(&asbd, 0, sizeof(asbd));
    asbd.mSampleRate = sr;
    asbd.mFormatID = kAudioFormatLinearPCM;
    asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    asbd.mBitsPerChannel = 32;
    asbd.mChannelsPerFrame = 2;
    asbd.mFramesPerPacket = 1;
    asbd.mBytesPerFrame = 8;
    asbd.mBytesPerPacket = 8;
    return asbd;
}

// MARK: - AudioServerPlugIn COM & Lifecycle

static HRESULT vc_QueryInterface(void *inDriver, REFIID inUUID, LPVOID *outInterface) {
    CFUUIDRef req = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    CFUUIDRef plugInIID = kAudioServerPlugInDriverInterfaceUUID;
    CFUUIDRef unknownIID = IUnknownUUID;
    Boolean match = CFEqual(req, plugInIID) || CFEqual(req, unknownIID);
    CFRelease(req);
    if (!match) {
        *outInterface = NULL;
        return E_NOINTERFACE;
    }
    pthread_mutex_lock(&gPlugInLock);
    ++gRefCount;
    pthread_mutex_unlock(&gPlugInLock);
    *outInterface = inDriver;
    return kAudioHardwareNoError;
}

static ULONG vc_AddRef(void *inDriver) {
    (void)inDriver;
    pthread_mutex_lock(&gPlugInLock);
    UInt32 r = ++gRefCount;
    pthread_mutex_unlock(&gPlugInLock);
    return r;
}

static ULONG vc_Release(void *inDriver) {
    (void)inDriver;
    pthread_mutex_lock(&gPlugInLock);
    UInt32 r = (gRefCount > 0) ? --gRefCount : 0;
    pthread_mutex_unlock(&gPlugInLock);
    return r;
}

static OSStatus vc_Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost) {
    (void)inDriver;
    gHost = inHost;
    for (size_t i = 0; i < 2; ++i) {
        vc_update_host_ticks(&gDevices[i]);
    }
    vc_open_ring(&gMicRing, 48000);
    vc_open_ring(&gMeetingRing, 48000);
    return kAudioHardwareNoError;
}

static OSStatus vc_CreateDevice(AudioServerPlugInDriverRef d, CFDictionaryRef desc, const AudioServerPlugInClientInfo *ci, AudioObjectID *outID) {
    (void)d; (void)desc; (void)ci; (void)outID;
    return kAudioHardwareUnsupportedOperationError;
}
static OSStatus vc_DestroyDevice(AudioServerPlugInDriverRef d, AudioObjectID id) {
    (void)d; (void)id;
    return kAudioHardwareUnsupportedOperationError;
}
static OSStatus vc_AddDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID id, const AudioServerPlugInClientInfo *ci) {
    (void)d; (void)id; (void)ci;
    return kAudioHardwareNoError;
}
static OSStatus vc_RemoveDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID id, const AudioServerPlugInClientInfo *ci) {
    (void)d; (void)id; (void)ci;
    return kAudioHardwareNoError;
}
static OSStatus vc_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef d, AudioObjectID id, UInt64 action, void *info) {
    (void)d; (void)action;
    VCDeviceState *dev = vc_find_device(id);
    if (dev && info) {
        Float64 sr = *(Float64 *)info;
        dev->sampleRate = sr;
        vc_update_host_ticks(dev);
        vc_open_ring(dev->ringHandle, (uint32_t)sr);
        free(info);
    }
    return kAudioHardwareNoError;
}
static OSStatus vc_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef d, AudioObjectID id, UInt64 action, void *info) {
    (void)d; (void)id; (void)action;
    if (info) free(info);
    return kAudioHardwareNoError;
}

// MARK: - Properties

enum {
    vcPropRunningSomewhere   = 'gone',
    vcPropBufferFrameSize    = 'fsiz',
    vcPropBufferFrameRange   = 'fsz#',
    vcPropStreamConfig       = 'slay',
    vcPropStreamUsage        = 'suse',
    vcPropActualSampleRate   = 'asrt'
};

typedef struct {
    void *mIOProc;
    UInt32 mNumberStreams;
    UInt32 mStreamIsOn[1];
} VCStreamUsage;

static Boolean vc_scope_matches(const VCDeviceState *dev, AudioObjectPropertyScope scope) {
    if (scope == kAudioObjectPropertyScopeGlobal) return true;
    if (dev->isInput && scope == kAudioObjectPropertyScopeInput) return true;
    if (!dev->isInput && scope == kAudioObjectPropertyScopeOutput) return true;
    return false;
}

static UInt32 vc_scope_channels(const VCDeviceState *dev, AudioObjectPropertyScope scope) {
    if (scope == kAudioObjectPropertyScopeGlobal) return 2;
    if (dev->isInput && scope == kAudioObjectPropertyScopeInput) return 2;
    if (!dev->isInput && scope == kAudioObjectPropertyScopeOutput) return 2;
    return 0;
}

static OSStatus vc_copy_u32(UInt32 val, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
    *(UInt32 *)outData = val;
    *outDataSize = sizeof(UInt32);
    return kAudioHardwareNoError;
}

static OSStatus vc_copy_f64(Float64 val, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    if (inDataSize < sizeof(Float64)) return kAudioHardwareBadPropertySizeError;
    *(Float64 *)outData = val;
    *outDataSize = sizeof(Float64);
    return kAudioHardwareNoError;
}

static OSStatus vc_copy_cfstr(CFStringRef val, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
    if (val) CFRetain(val);
    *(CFStringRef *)outData = val;
    *outDataSize = sizeof(CFStringRef);
    return kAudioHardwareNoError;
}

static Boolean vc_plugIn_has_property(AudioObjectPropertySelector sel) {
    switch (sel) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioObjectPropertyControlList:
        case kAudioObjectPropertyCustomPropertyInfoList:
        case kAudioPlugInPropertyBundleID:
        case kAudioPlugInPropertyDeviceList:
        case kAudioPlugInPropertyTranslateUIDToDevice:
        case kAudioPlugInPropertyBoxList:
        case kAudioPlugInPropertyTranslateUIDToBox:
        case kAudioPlugInPropertyClockDeviceList:
        case kAudioPlugInPropertyTranslateUIDToClockDevice:
        case kAudioPlugInPropertyResourceBundle:
            return true;
        default:
            return false;
    }
}

static Boolean vc_device_has_property(AudioObjectPropertySelector sel) {
    switch (sel) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioObjectPropertyControlList:
        case kAudioObjectPropertyCustomPropertyInfoList:
        case kAudioDevicePropertyDeviceUID:
        case kAudioDevicePropertyModelUID:
        case kAudioDevicePropertyTransportType:
        case kAudioDevicePropertyRelatedDevices:
        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceIsRunning:
        case vcPropRunningSomewhere:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertyStreams:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyNominalSampleRate:
        case kAudioDevicePropertyAvailableNominalSampleRates:
        case vcPropActualSampleRate:
        case kAudioDevicePropertyZeroTimeStampPeriod:
        case kAudioDevicePropertyClockAlgorithm:
        case kAudioDevicePropertyClockIsStable:
        case kAudioDevicePropertyIsHidden:
        case kAudioDevicePropertyPreferredChannelsForStereo:
        case kAudioDevicePropertyPreferredChannelLayout:
        case vcPropBufferFrameSize:
        case vcPropBufferFrameRange:
        case vcPropStreamConfig:
        case vcPropStreamUsage:
            return true;
        default:
            return false;
    }
}

static Boolean vc_stream_has_property(AudioObjectPropertySelector sel) {
    switch (sel) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioObjectPropertyControlList:
        case kAudioObjectPropertyCustomPropertyInfoList:
        case kAudioStreamPropertyIsActive:
        case kAudioStreamPropertyDirection:
        case kAudioStreamPropertyTerminalType:
        case kAudioStreamPropertyStartingChannel:
        case kAudioStreamPropertyLatency:
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyPhysicalFormat:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            return true;
        default:
            return false;
    }
}

static Boolean vc_HasProperty(AudioServerPlugInDriverRef d, AudioObjectID objID, pid_t pid, const AudioObjectPropertyAddress *addr) {
    (void)d; (void)pid;
    if (!addr) return false;
    if (objID == kObjectID_PlugIn) {
        return vc_plugIn_has_property(addr->mSelector);
    }
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return false;
    if (objID == dev->deviceID) {
        return vc_device_has_property(addr->mSelector);
    }
    if (objID == dev->streamID) {
        return vc_stream_has_property(addr->mSelector);
    }
    return false;
}

static OSStatus vc_IsPropertySettable(AudioServerPlugInDriverRef d, AudioObjectID objID, pid_t pid, const AudioObjectPropertyAddress *addr, Boolean *outSettable) {
    (void)d; (void)pid;
    if (!addr || !outSettable) return kAudioHardwareIllegalOperationError;
    *outSettable = false;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareNoError;
    if (objID == dev->deviceID &&
        (addr->mSelector == kAudioDevicePropertyNominalSampleRate || addr->mSelector == vcPropBufferFrameSize)) {
        *outSettable = true;
    } else if (objID == dev->streamID &&
               (addr->mSelector == kAudioStreamPropertyVirtualFormat || addr->mSelector == kAudioStreamPropertyPhysicalFormat)) {
        *outSettable = true;
    }
    return kAudioHardwareNoError;
}

static OSStatus vc_GetPropertyDataSize(AudioServerPlugInDriverRef d, AudioObjectID objID, pid_t pid, const AudioObjectPropertyAddress *addr, UInt32 inQSize, const void *inQ, UInt32 *outSize) {
    (void)d; (void)pid; (void)inQSize; (void)inQ;
    if (!addr || !outSize) return kAudioHardwareIllegalOperationError;
    if (objID == kObjectID_PlugIn) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
            case kAudioPlugInPropertyTranslateUIDToDevice:
            case kAudioPlugInPropertyTranslateUIDToBox:
            case kAudioPlugInPropertyTranslateUIDToClockDevice:
                *outSize = sizeof(UInt32); return kAudioHardwareNoError;
            case kAudioObjectPropertyName:
            case kAudioObjectPropertyManufacturer:
            case kAudioPlugInPropertyBundleID:
            case kAudioPlugInPropertyResourceBundle:
                *outSize = sizeof(CFStringRef); return kAudioHardwareNoError;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioPlugInPropertyDeviceList:
                *outSize = 2 * sizeof(AudioObjectID); return kAudioHardwareNoError;
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
            case kAudioPlugInPropertyBoxList:
            case kAudioPlugInPropertyClockDeviceList:
                *outSize = 0; return kAudioHardwareNoError;
            default:
                vc_log_unknown("GetPropertyDataSize", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareBadObjectError;

    if (objID == dev->deviceID) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
            case kAudioDevicePropertyTransportType:
            case kAudioDevicePropertyRelatedDevices:
            case kAudioDevicePropertyClockDomain:
            case kAudioDevicePropertyDeviceIsAlive:
            case kAudioDevicePropertyDeviceIsRunning:
            case vcPropRunningSomewhere:
            case kAudioDevicePropertyDeviceCanBeDefaultDevice:
            case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
            case kAudioDevicePropertyLatency:
            case kAudioDevicePropertySafetyOffset:
            case kAudioDevicePropertyZeroTimeStampPeriod:
            case kAudioDevicePropertyClockAlgorithm:
            case kAudioDevicePropertyClockIsStable:
            case kAudioDevicePropertyIsHidden:
            case vcPropBufferFrameSize:
                *outSize = sizeof(UInt32); return kAudioHardwareNoError;
            case kAudioObjectPropertyName:
            case kAudioObjectPropertyManufacturer:
            case kAudioDevicePropertyDeviceUID:
            case kAudioDevicePropertyModelUID:
                *outSize = sizeof(CFStringRef); return kAudioHardwareNoError;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioDevicePropertyStreams:
                *outSize = vc_scope_matches(dev, addr->mScope) ? sizeof(AudioObjectID) : 0;
                return kAudioHardwareNoError;
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
                *outSize = 0; return kAudioHardwareNoError;
            case kAudioDevicePropertyNominalSampleRate:
            case vcPropActualSampleRate:
                *outSize = sizeof(Float64); return kAudioHardwareNoError;
            case kAudioDevicePropertyAvailableNominalSampleRates:
                *outSize = 2 * sizeof(AudioValueRange); return kAudioHardwareNoError;
            case vcPropBufferFrameRange:
                *outSize = sizeof(AudioValueRange); return kAudioHardwareNoError;
            case kAudioDevicePropertyPreferredChannelsForStereo:
                *outSize = 2 * sizeof(UInt32); return kAudioHardwareNoError;
            case kAudioDevicePropertyPreferredChannelLayout:
                *outSize = (UInt32)(offsetof(AudioChannelLayout, mChannelDescriptions) + 2 * sizeof(AudioChannelDescription));
                return kAudioHardwareNoError;
            case vcPropStreamConfig: {
                UInt32 nBuf = (vc_scope_channels(dev, addr->mScope) > 0) ? 1 : 0;
                *outSize = (UInt32)(offsetof(AudioBufferList, mBuffers) + nBuf * sizeof(AudioBuffer));
                return kAudioHardwareNoError;
            }
            case vcPropStreamUsage: {
                UInt32 nStr = vc_scope_matches(dev, addr->mScope) ? 1 : 0;
                *outSize = (UInt32)(offsetof(VCStreamUsage, mStreamIsOn) + nStr * sizeof(UInt32));
                return kAudioHardwareNoError;
            }
            default:
                vc_log_unknown("GetPropertyDataSize", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }

    if (objID == dev->streamID) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
            case kAudioStreamPropertyIsActive:
            case kAudioStreamPropertyDirection:
            case kAudioStreamPropertyTerminalType:
            case kAudioStreamPropertyStartingChannel:
            case kAudioStreamPropertyLatency:
                *outSize = sizeof(UInt32); return kAudioHardwareNoError;
            case kAudioObjectPropertyName:
            case kAudioObjectPropertyManufacturer:
                *outSize = sizeof(CFStringRef); return kAudioHardwareNoError;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
                *outSize = 0; return kAudioHardwareNoError;
            case kAudioStreamPropertyVirtualFormat:
            case kAudioStreamPropertyPhysicalFormat:
                *outSize = sizeof(AudioStreamBasicDescription); return kAudioHardwareNoError;
            case kAudioStreamPropertyAvailableVirtualFormats:
            case kAudioStreamPropertyAvailablePhysicalFormats:
                *outSize = 2 * sizeof(AudioStreamRangedDescription); return kAudioHardwareNoError;
            default:
                vc_log_unknown("GetPropertyDataSize", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }
    return kAudioHardwareBadObjectError;
}

static OSStatus vc_GetPropertyData(AudioServerPlugInDriverRef d, AudioObjectID objID, pid_t pid, const AudioObjectPropertyAddress *addr, UInt32 inQSize, const void *inQ, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    (void)d; (void)pid;
    if (!addr || !outDataSize) return kAudioHardwareIllegalOperationError;
    if (objID == kObjectID_PlugIn) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
                return vc_copy_u32(kAudioObjectClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyClass:
                return vc_copy_u32(kAudioPlugInClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwner:
                return vc_copy_u32(kAudioObjectUnknown, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyName:
                return vc_copy_cfstr(CFSTR("VoiceClone Audio Driver"), inDataSize, outDataSize, outData);
            case kAudioObjectPropertyManufacturer:
                return vc_copy_cfstr(CFSTR("VoiceClone Live"), inDataSize, outDataSize, outData);
            case kAudioPlugInPropertyBundleID:
                return vc_copy_cfstr(kPlugInBundleID, inDataSize, outDataSize, outData);
            case kAudioPlugInPropertyResourceBundle:
                return vc_copy_cfstr(CFSTR(""), inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwnedObjects:
            case kAudioPlugInPropertyDeviceList: {
                if (inDataSize < 2 * sizeof(AudioObjectID) || !outData) return kAudioHardwareBadPropertySizeError;
                AudioObjectID *ids = (AudioObjectID *)outData;
                ids[0] = kObjectID_MicDevice;
                ids[1] = kObjectID_MeetingDevice;
                *outDataSize = 2 * sizeof(AudioObjectID);
                return kAudioHardwareNoError;
            }
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
            case kAudioPlugInPropertyBoxList:
            case kAudioPlugInPropertyClockDeviceList:
                *outDataSize = 0;
                return kAudioHardwareNoError;
            case kAudioPlugInPropertyTranslateUIDToDevice: {
                if (inQSize < sizeof(CFStringRef) || !inQ || inDataSize < sizeof(AudioObjectID) || !outData) {
                    return kAudioHardwareBadPropertySizeError;
                }
                CFStringRef uid = *(const CFStringRef *)inQ;
                AudioObjectID res = kAudioObjectUnknown;
                if (uid) {
                    for (size_t i = 0; i < 2; ++i) {
                        if (CFEqual(uid, gDevices[i].uid)) { res = gDevices[i].deviceID; break; }
                    }
                }
                *(AudioObjectID *)outData = res;
                *outDataSize = sizeof(AudioObjectID);
                return kAudioHardwareNoError;
            }
            case kAudioPlugInPropertyTranslateUIDToBox:
            case kAudioPlugInPropertyTranslateUIDToClockDevice:
                return vc_copy_u32(kAudioObjectUnknown, inDataSize, outDataSize, outData);
            default:
                vc_log_unknown("GetPropertyData", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }

    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareBadObjectError;

    if (objID == dev->deviceID) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
                return vc_copy_u32(kAudioObjectClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyClass:
                return vc_copy_u32(kAudioDeviceClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwner:
                return vc_copy_u32(kObjectID_PlugIn, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyName:
                return vc_copy_cfstr(dev->name, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyManufacturer:
                return vc_copy_cfstr(CFSTR("VoiceClone Live"), inDataSize, outDataSize, outData);
            case kAudioDevicePropertyDeviceUID:
                return vc_copy_cfstr(dev->uid, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyModelUID:
                return vc_copy_cfstr(CFSTR("com.voiceclone.driver.model"), inDataSize, outDataSize, outData);
            case kAudioDevicePropertyTransportType:
                /* Report USB transport so Mac Catalyst / VoIP apps (e.g., WhatsApp) expose the device instead of skipping Virtual transport. */
                return vc_copy_u32(kAudioDeviceTransportTypeUSB, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyRelatedDevices:
                return vc_copy_u32(dev->deviceID, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyClockDomain:
                return vc_copy_u32(0, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyDeviceIsAlive:
                return vc_copy_u32(1, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyDeviceIsRunning:
            case vcPropRunningSomewhere:
                return vc_copy_u32((atomic_load(&dev->ioCount) > 0) ? 1 : 0, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyDeviceCanBeDefaultDevice:
                return vc_copy_u32(vc_scope_matches(dev, addr->mScope) ? 1 : 0, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
                return vc_copy_u32((!dev->isInput && vc_scope_matches(dev, addr->mScope)) ? 1 : 0, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyLatency:
            case kAudioDevicePropertySafetyOffset:
            case kAudioDevicePropertyIsHidden:
                return vc_copy_u32(0, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyZeroTimeStampPeriod:
                return vc_copy_u32(VC_ZERO_TS_PERIOD, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyClockAlgorithm:
                return vc_copy_u32(kAudioDeviceClockAlgorithmRaw, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyClockIsStable:
                return vc_copy_u32(1, inDataSize, outDataSize, outData);
            case vcPropBufferFrameSize:
                return vc_copy_u32(dev->bufferFrameSize, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwnedObjects:
            case kAudioDevicePropertyStreams:
                if (vc_scope_matches(dev, addr->mScope)) {
                    return vc_copy_u32(dev->streamID, inDataSize, outDataSize, outData);
                }
                *outDataSize = 0;
                return kAudioHardwareNoError;
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
                *outDataSize = 0;
                return kAudioHardwareNoError;
            case kAudioDevicePropertyNominalSampleRate:
            case vcPropActualSampleRate:
                return vc_copy_f64(dev->sampleRate, inDataSize, outDataSize, outData);
            case kAudioDevicePropertyAvailableNominalSampleRates: {
                if (inDataSize < 2 * sizeof(AudioValueRange) || !outData) return kAudioHardwareBadPropertySizeError;
                AudioValueRange *r = (AudioValueRange *)outData;
                r[0] = (AudioValueRange){ 16000.0, 16000.0 };
                r[1] = (AudioValueRange){ 48000.0, 48000.0 };
                *outDataSize = 2 * sizeof(AudioValueRange);
                return kAudioHardwareNoError;
            }
            case vcPropBufferFrameRange: {
                if (inDataSize < sizeof(AudioValueRange) || !outData) return kAudioHardwareBadPropertySizeError;
                *(AudioValueRange *)outData = (AudioValueRange){ 64.0, 4096.0 };
                *outDataSize = sizeof(AudioValueRange);
                return kAudioHardwareNoError;
            }
            case kAudioDevicePropertyPreferredChannelsForStereo: {
                if (inDataSize < 2 * sizeof(UInt32) || !outData) return kAudioHardwareBadPropertySizeError;
                UInt32 *ch = (UInt32 *)outData;
                ch[0] = 1; ch[1] = 2;
                *outDataSize = 2 * sizeof(UInt32);
                return kAudioHardwareNoError;
            }
            case kAudioDevicePropertyPreferredChannelLayout: {
                UInt32 needed = (UInt32)(offsetof(AudioChannelLayout, mChannelDescriptions) + 2 * sizeof(AudioChannelDescription));
                if (inDataSize < needed || !outData) return kAudioHardwareBadPropertySizeError;
                AudioChannelLayout *cl = (AudioChannelLayout *)outData;
                memset(cl, 0, needed);
                cl->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
                cl->mNumberChannelDescriptions = 2;
                cl->mChannelDescriptions[0].mChannelLabel = kAudioChannelLabel_Left;
                cl->mChannelDescriptions[1].mChannelLabel = kAudioChannelLabel_Right;
                *outDataSize = needed;
                return kAudioHardwareNoError;
            }
            case vcPropStreamConfig: {
                UInt32 ch = vc_scope_channels(dev, addr->mScope);
                UInt32 nBuf = (ch > 0) ? 1 : 0;
                UInt32 needed = (UInt32)(offsetof(AudioBufferList, mBuffers) + nBuf * sizeof(AudioBuffer));
                if (inDataSize < needed || !outData) return kAudioHardwareBadPropertySizeError;
                AudioBufferList *abl = (AudioBufferList *)outData;
                memset(abl, 0, needed);
                abl->mNumberBuffers = nBuf;
                if (nBuf > 0) {
                    abl->mBuffers[0].mNumberChannels = ch;
                }
                *outDataSize = needed;
                return kAudioHardwareNoError;
            }
            case vcPropStreamUsage: {
                UInt32 nStr = vc_scope_matches(dev, addr->mScope) ? 1 : 0;
                UInt32 needed = (UInt32)(offsetof(VCStreamUsage, mStreamIsOn) + nStr * sizeof(UInt32));
                if (inDataSize < needed || !outData) return kAudioHardwareBadPropertySizeError;
                VCStreamUsage *su = (VCStreamUsage *)outData;
                su->mIOProc = (inQSize >= sizeof(void *) && inQ) ? *(void *const *)inQ : NULL;
                su->mNumberStreams = nStr;
                if (nStr > 0) su->mStreamIsOn[0] = 1;
                *outDataSize = needed;
                return kAudioHardwareNoError;
            }
            default:
                vc_log_unknown("GetPropertyData", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }

    if (objID == dev->streamID) {
        switch (addr->mSelector) {
            case kAudioObjectPropertyBaseClass:
                return vc_copy_u32(kAudioObjectClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyClass:
                return vc_copy_u32(kAudioStreamClassID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwner:
                return vc_copy_u32(dev->deviceID, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyName:
                return vc_copy_cfstr(dev->name, inDataSize, outDataSize, outData);
            case kAudioObjectPropertyManufacturer:
                return vc_copy_cfstr(CFSTR("VoiceClone Live"), inDataSize, outDataSize, outData);
            case kAudioObjectPropertyOwnedObjects:
            case kAudioObjectPropertyControlList:
            case kAudioObjectPropertyCustomPropertyInfoList:
                *outDataSize = 0;
                return kAudioHardwareNoError;
            case kAudioStreamPropertyIsActive:
                return vc_copy_u32(1, inDataSize, outDataSize, outData);
            case kAudioStreamPropertyDirection:
                return vc_copy_u32(dev->isInput ? 1 : 0, inDataSize, outDataSize, outData);
            case kAudioStreamPropertyTerminalType:
                return vc_copy_u32(dev->isInput ? kAudioStreamTerminalTypeMicrophone : kAudioStreamTerminalTypeSpeaker, inDataSize, outDataSize, outData);
            case kAudioStreamPropertyStartingChannel:
                return vc_copy_u32(1, inDataSize, outDataSize, outData);
            case kAudioStreamPropertyLatency:
                return vc_copy_u32(0, inDataSize, outDataSize, outData);
            case kAudioStreamPropertyVirtualFormat:
            case kAudioStreamPropertyPhysicalFormat:
                if (inDataSize < sizeof(AudioStreamBasicDescription) || !outData) return kAudioHardwareBadPropertySizeError;
                *(AudioStreamBasicDescription *)outData = vc_make_asbd(dev->sampleRate);
                *outDataSize = sizeof(AudioStreamBasicDescription);
                return kAudioHardwareNoError;
            case kAudioStreamPropertyAvailableVirtualFormats:
            case kAudioStreamPropertyAvailablePhysicalFormats: {
                if (inDataSize < 2 * sizeof(AudioStreamRangedDescription) || !outData) return kAudioHardwareBadPropertySizeError;
                AudioStreamRangedDescription *r = (AudioStreamRangedDescription *)outData;
                r[0].mFormat = vc_make_asbd(16000.0);
                r[0].mSampleRateRange = (AudioValueRange){ 16000.0, 16000.0 };
                r[1].mFormat = vc_make_asbd(48000.0);
                r[1].mSampleRateRange = (AudioValueRange){ 48000.0, 48000.0 };
                *outDataSize = 2 * sizeof(AudioStreamRangedDescription);
                return kAudioHardwareNoError;
            }
            default:
                vc_log_unknown("GetPropertyData", objID, addr);
                return kAudioHardwareUnknownPropertyError;
        }
    }
    return kAudioHardwareBadObjectError;
}

static OSStatus vc_SetPropertyData(AudioServerPlugInDriverRef d, AudioObjectID objID, pid_t pid, const AudioObjectPropertyAddress *addr, UInt32 inQSize, const void *inQ, UInt32 inDataSize, const void *inData) {
    (void)d; (void)pid; (void)inQSize; (void)inQ;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev || !addr || !inData) return kAudioHardwareBadObjectError;
    if (objID == dev->deviceID && addr->mSelector == vcPropBufferFrameSize && inDataSize >= sizeof(UInt32)) {
        UInt32 frames = *(const UInt32 *)inData;
        if (frames < 64) frames = 64;
        if (frames > 4096) frames = 4096;
        dev->bufferFrameSize = frames;
        return kAudioHardwareNoError;
    }
    Float64 newSR = 0.0;
    if (objID == dev->deviceID && addr->mSelector == kAudioDevicePropertyNominalSampleRate && inDataSize >= sizeof(Float64)) {
        newSR = *(const Float64 *)inData;
    } else if (objID == dev->streamID && (addr->mSelector == kAudioStreamPropertyVirtualFormat || addr->mSelector == kAudioStreamPropertyPhysicalFormat) && inDataSize >= sizeof(AudioStreamBasicDescription)) {
        newSR = ((const AudioStreamBasicDescription *)inData)->mSampleRate;
    } else {
        vc_log_unknown("SetPropertyData", objID, addr);
        return kAudioHardwareUnsupportedOperationError;
    }
    if (newSR != 16000.0 && newSR != 48000.0) {
        return kAudioDeviceUnsupportedFormatError;
    }
    if (gHost && newSR != dev->sampleRate) {
        Float64 *ctx = (Float64 *)malloc(sizeof(Float64));
        if (!ctx) return kAudioHardwareUnspecifiedError;
        *ctx = newSR;
        return gHost->RequestDeviceConfigurationChange(gHost, dev->deviceID, 0, ctx);
    }
    return kAudioHardwareNoError;
}

// MARK: - IO Operations

static void vc_set_ring_io_count(VCRingHandle *h, uint32_t count) {
    pthread_mutex_lock(&h->lock);
    if (h->ring != NULL) {
        atomic_store(&h->ring->reserved, count);
    }
    pthread_mutex_unlock(&h->lock);
}

static OSStatus vc_StartIO(AudioServerPlugInDriverRef d, AudioObjectID objID, UInt32 clientID) {
    (void)d; (void)clientID;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareBadObjectError;
    uint32_t next = atomic_fetch_add(&dev->ioCount, 1) + 1;
    if (next == 1) {
        if (dev->hostTicksPerFrame <= 0.0) {
            vc_update_host_ticks(dev);
        }
        dev->numberTimeStamps = 0;
        dev->anchorHostTime = mach_absolute_time();
        vc_open_ring(dev->ringHandle, (uint32_t)dev->sampleRate);
        if (dev->ringHandle && dev->ringHandle->ring) {
            VCRingBuffer *rb = dev->ringHandle->ring;
            uint64_t w = atomic_load_explicit(&rb->write_pos, memory_order_acquire);
            uint64_t r = atomic_load_explicit(&rb->read_pos, memory_order_relaxed);
            uint64_t maxPreRoll = dev->isInput ? ((uint64_t)rb->sample_rate / 4U) : 0U;
            if (r > w || (w - r) > maxPreRoll) {
                uint64_t nextRead = (w > maxPreRoll) ? (w - maxPreRoll) : 0U;
                atomic_store_explicit(&rb->read_pos, nextRead, memory_order_release);
            }
        }
    }
    vc_set_ring_io_count(dev->ringHandle, next);
    return kAudioHardwareNoError;
}

static OSStatus vc_StopIO(AudioServerPlugInDriverRef d, AudioObjectID objID, UInt32 clientID) {
    (void)d; (void)clientID;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareBadObjectError;
    uint32_t prev = atomic_load(&dev->ioCount);
    while (prev > 0 && !atomic_compare_exchange_weak(&dev->ioCount, &prev, prev - 1)) {
    }
    uint32_t nowCount = (prev > 0) ? (prev - 1) : 0;
    vc_set_ring_io_count(dev->ringHandle, nowCount);
    return kAudioHardwareNoError;
}

static OSStatus vc_GetZeroTimeStamp(AudioServerPlugInDriverRef d, AudioObjectID objID, UInt32 clientID, Float64 *outSampleTime, UInt64 *outHostTime, UInt64 *outSeed) {
    (void)d; (void)clientID;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev) return kAudioHardwareBadObjectError;
    UInt64 now = mach_absolute_time();
    Float64 ticksPerPeriod = dev->hostTicksPerFrame * (Float64)VC_ZERO_TS_PERIOD;
    UInt64 elapsed = (now >= dev->anchorHostTime) ? (now - dev->anchorHostTime) : 0;
    UInt64 periods = (ticksPerPeriod > 0.0) ? (UInt64)((Float64)elapsed / ticksPerPeriod) : 0;
    dev->numberTimeStamps = periods;
    *outSampleTime = (Float64)(periods * (UInt64)VC_ZERO_TS_PERIOD);
    *outHostTime = dev->anchorHostTime + (UInt64)((Float64)periods * ticksPerPeriod);
    *outSeed = 1;
    return kAudioHardwareNoError;
}

static OSStatus vc_WillDoIOOperation(AudioServerPlugInDriverRef d, AudioObjectID objID, UInt32 clientID, UInt32 opID, Boolean *outWillDo, Boolean *outWillDoInPlace) {
    (void)d; (void)clientID;
    VCDeviceState *dev = vc_find_device(objID);
    *outWillDo = false;
    *outWillDoInPlace = true;
    if (!dev) return kAudioHardwareBadObjectError;
    if (dev->isInput && opID == kAudioServerPlugInIOOperationReadInput) {
        *outWillDo = true;
    } else if (!dev->isInput && opID == kAudioServerPlugInIOOperationWriteMix) {
        *outWillDo = true;
    }
    return kAudioHardwareNoError;
}

static OSStatus vc_BeginIOOperation(AudioServerPlugInDriverRef d, AudioObjectID id, UInt32 cid, UInt32 op, UInt32 sz, const AudioServerPlugInIOCycleInfo *info) {
    (void)d; (void)id; (void)cid; (void)op; (void)sz; (void)info;
    return kAudioHardwareNoError;
}

static OSStatus vc_DoIOOperation(AudioServerPlugInDriverRef d, AudioObjectID objID, AudioObjectID streamID, UInt32 clientID, UInt32 opID, UInt32 ioBufferFrameSize, const AudioServerPlugInIOCycleInfo *info, void *ioMainBuffer, void *ioSecondaryBuffer) {
    (void)d; (void)streamID; (void)clientID; (void)info; (void)ioSecondaryBuffer;
    VCDeviceState *dev = vc_find_device(objID);
    if (!dev || !ioMainBuffer) return kAudioHardwareNoError;
    if (dev->isInput && opID == kAudioServerPlugInIOOperationReadInput) {
        vc_ring_read_stereo(dev->ringHandle, (float *)ioMainBuffer, ioBufferFrameSize);
    } else if (!dev->isInput && opID == kAudioServerPlugInIOOperationWriteMix) {
        vc_ring_write_stereo(dev->ringHandle, (const float *)ioMainBuffer, ioBufferFrameSize);
    }
    return kAudioHardwareNoError;
}

static OSStatus vc_EndIOOperation(AudioServerPlugInDriverRef d, AudioObjectID id, UInt32 cid, UInt32 op, UInt32 sz, const AudioServerPlugInIOCycleInfo *info) {
    (void)d; (void)id; (void)cid; (void)op; (void)sz; (void)info;
    return kAudioHardwareNoError;
}

static AudioServerPlugInDriverInterface gDriverInterface = {
    NULL,
    vc_QueryInterface,
    vc_AddRef,
    vc_Release,
    vc_Initialize,
    vc_CreateDevice,
    vc_DestroyDevice,
    vc_AddDeviceClient,
    vc_RemoveDeviceClient,
    vc_PerformDeviceConfigurationChange,
    vc_AbortDeviceConfigurationChange,
    vc_HasProperty,
    vc_IsPropertySettable,
    vc_GetPropertyDataSize,
    vc_GetPropertyData,
    vc_SetPropertyData,
    vc_StartIO,
    vc_StopIO,
    vc_GetZeroTimeStamp,
    vc_WillDoIOOperation,
    vc_BeginIOOperation,
    vc_DoIOOperation,
    vc_EndIOOperation,
};

static AudioServerPlugInDriverInterface *gDriverInterfacePtr = &gDriverInterface;

__attribute__((visibility("default")))
void *VoiceCloneAudioDriverFactory(CFAllocatorRef inAllocator, CFUUIDRef inRequestedTypeUUID) {
    (void)inAllocator;
    CFUUIDRef pluginType = kAudioServerPlugInTypeUUID;
    if (CFEqual(inRequestedTypeUUID, pluginType)) {
        return &gDriverInterfacePtr;
    }
    return NULL;
}
