import AudioToolbox
import CoreAudio
import Foundation

// MARK: - Signal & Thread-Safe PCM Ring Queue

private var gShouldStop = false

private func handleSignal(_ sig: Int32) {
    gShouldStop = true
}

final class PCMByteQueue {
    private var lock = pthread_mutex_t()
    private let storage: UnsafeMutablePointer<UInt8>
    private let capacity: Int
    private var readIndex = 0
    private var used = 0
    private var oddByte: UInt8?
    private var droppedBytes = 0

    init(maxBytes: Int = 48_000) {
        capacity = maxBytes & ~1
        storage = .allocate(capacity: capacity)
        pthread_mutex_init(&lock, nil)
    }

    deinit {
        pthread_mutex_destroy(&lock)
        storage.deallocate()
    }

    private func copyIn(_ src: UnsafePointer<UInt8>, _ count: Int) {
        let excess = max(0, used + count - capacity)
        readIndex = (readIndex + excess) % capacity
        used -= excess
        droppedBytes += excess
        let writeIndex = (readIndex + used) % capacity
        let first = min(count, capacity - writeIndex)
        memcpy(storage + writeIndex, src, first)
        if count > first { memcpy(storage, src + first, count - first) }
        used += count
    }

    func push(_ raw: UnsafeRawBufferPointer) {
        guard var src = raw.bindMemory(to: UInt8.self).baseAddress, !raw.isEmpty else { return }
        pthread_mutex_lock(&lock)
        defer { pthread_mutex_unlock(&lock) }
        var count = raw.count
        if let byte = oddByte {
            var pair = UInt16(byte) | UInt16(src[0]) << 8
            withUnsafeBytes(of: &pair) { copyIn($0.bindMemory(to: UInt8.self).baseAddress!, 2) }
            src += 1
            count -= 1
            oddByte = nil
        }
        if count % 2 != 0 {
            count -= 1
            oddByte = src[count]
        }
        let excess = max(0, count - capacity)
        droppedBytes += excess
        copyIn(src + excess, count - excess)
    }

    func popFrame(into dst: UnsafeMutableRawPointer, bytes: Int) -> Bool {
        pthread_mutex_lock(&lock)
        defer { pthread_mutex_unlock(&lock) }
        guard used >= bytes else { return false }
        let first = min(bytes, capacity - readIndex)
        memcpy(dst, storage + readIndex, first)
        if bytes > first { memcpy(dst + first, storage, bytes - first) }
        readIndex = (readIndex + bytes) % capacity
        used -= bytes
        return true
    }

    private var speakerPriming = true
    private var speakerEnded = false
    private var speakerGaps = 0
    private var speakerRecovering = false
    private var speakerRecoveryWaitBytes = 0
    private var speakerMaxRecoveryWaitBytes = 0
    private var speakerFirstPCMUnixMs: Int64 = 0

    func endSpeakerTurn() {
        pthread_mutex_lock(&lock)
        speakerEnded = true
        pthread_mutex_unlock(&lock)
    }

    func renderSpeaker(_ dst: UnsafeMutableRawPointer, bytes: Int, sampleRate: Int = 24_000) {
        // Use fast ~100 ms initial turn prebuffer for snappy live translation onset,
        // and ~190 ms smoothing only when recovering from mid-turn network underrun.
        let initialRefillBytes = sampleRate * 20 / 100
        let recoveryRefillBytes = sampleRate * 38 / 100
        let initialMaxWaitBytes = sampleRate * 24 / 100
        let recoveryMaxWaitBytes = sampleRate * 44 / 100
        pthread_mutex_lock(&lock)
        defer { pthread_mutex_unlock(&lock) }
        let targetRefill = speakerRecovering ? recoveryRefillBytes : initialRefillBytes
        let maxWaitBytes = speakerRecovering ? recoveryMaxWaitBytes : initialMaxWaitBytes
        if speakerPriming && !speakerEnded && used < targetRefill
            && !(used > 0 && speakerRecoveryWaitBytes >= maxWaitBytes) {
            if used > 0 {
                speakerRecoveryWaitBytes += bytes
            }
            memset(dst, 0, bytes)
            return
        }
        if speakerRecovering && used > 0 {
            speakerMaxRecoveryWaitBytes = max(speakerMaxRecoveryWaitBytes, speakerRecoveryWaitBytes)
        }
        speakerRecoveryWaitBytes = 0
        let count = min(used, bytes)
        if count > 0 && speakerFirstPCMUnixMs == 0 {
            speakerFirstPCMUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
        }
        let first = min(count, capacity - readIndex)
        memcpy(dst, storage + readIndex, first)
        if count > first { memcpy(dst + first, storage, count - first) }
        readIndex = (readIndex + count) % capacity
        used -= count
        if count < bytes { memset(dst + count, 0, bytes - count) }
        if used == 0 && speakerEnded {
            speakerPriming = true
            speakerEnded = false
            speakerRecovering = false
        } else if count < bytes {
            if !speakerPriming || count > 0 {
                speakerGaps += 1
                speakerRecovering = true
            }
            speakerPriming = true
        } else {
            speakerPriming = false
        }
    }

    func reportDrops(_ label: String, sampleRate: Int = 24_000) {
        pthread_mutex_lock(&lock)
        let dropped = droppedBytes
        let gaps = speakerGaps
        let recoveryWaitMs = speakerMaxRecoveryWaitBytes * 500 / sampleRate
        let firstPCM = speakerFirstPCMUnixMs
        pthread_mutex_unlock(&lock)
        fputs("Native audio \(label): dropped_samples=\(dropped / 2) gaps=\(gaps) max_buffered_recovery_wait_ms=\(recoveryWaitMs) first_pcm_unix_ms=\(firstPCM)\n", stderr)
    }

    func popFloats(count: Int, into dst: UnsafeMutablePointer<Float>) {
        guard pthread_mutex_trylock(&lock) == 0 else {
            dst.update(repeating: 0, count: count)
            return
        }
        let samples = min(count, used / 2)
        for i in 0..<samples {
            let bits = UInt16(storage[readIndex]) | UInt16(storage[readIndex + 1]) << 8
            dst[i] = Float(Int16(bitPattern: bits)) / 32768.0
            readIndex += 2
            if readIndex == capacity { readIndex = 0 }
        }
        used -= samples * 2
        pthread_mutex_unlock(&lock)
        if samples < count { (dst + samples).update(repeating: 0, count: count - samples) }
    }
}

private func startCaptureWriter(_ queue: PCMByteQueue) {
    Thread.detachNewThread {
        var bytes = [UInt8](repeating: 0, count: 960)
        while !gShouldStop {
            let ready = bytes.withUnsafeMutableBytes { queue.popFrame(into: $0.baseAddress!, bytes: $0.count) }
            if ready {
                bytes.withUnsafeBytes { writeStdoutFrame($0) }
            } else {
                Thread.sleep(forTimeInterval: 0.002)
            }
        }
    }
}

private func writeStdoutFrame(_ raw: UnsafeRawBufferPointer) {
    do {
        guard let base = raw.baseAddress else { return }
        var written = 0
        while written < raw.count && !gShouldStop {
            let n = Darwin.write(STDOUT_FILENO, base.advanced(by: written), raw.count - written)
            if n <= 0 {
                if errno == EINTR { continue }
                gShouldStop = true
                return
            }
            written += n
        }
    }
}

private func readStdinExact(_ ptr: UnsafeMutableRawPointer, _ count: Int) -> Bool {
    var offset = 0
    while offset < count && !gShouldStop {
        let n = Darwin.read(STDIN_FILENO, ptr + offset, count - offset)
        if n < 0 && errno == EINTR { continue }
        if n <= 0 { return false }
        offset += n
    }
    return offset == count
}

private func startStdinMonitor(playbackQueue: PCMByteQueue?) {
    Thread.detachNewThread {
        if let q = playbackQueue {
            var buf = [UInt8](repeating: 0, count: 192_000)
            var length: UInt32 = 0
            while withUnsafeMutableBytes(of: &length, { readStdinExact($0.baseAddress!, $0.count) }) {
                let n = Int(UInt32(littleEndian: length))
                if n == 0 { q.endSpeakerTurn(); continue }
                guard n <= buf.count && n % 2 == 0,
                      buf.withUnsafeMutableBytes({ readStdinExact($0.baseAddress!, n) }) else { break }
                buf.withUnsafeBytes { q.push(UnsafeRawBufferPointer(rebasing: $0[..<n])) }
            }
        } else {
            var byte: UInt8 = 0
            while readStdinExact(&byte, 1) {}
        }
        gShouldStop = true
    }
}

// MARK: - Mode 1: Apple VoiceProcessingIO Full-Duplex (Hardware AEC)

final class VPIOContext {
    var audioUnit: AudioUnit!
    let playbackQueue = PCMByteQueue(maxBytes: 384_000)
    var renderScratch: UnsafeMutablePointer<Float>
    var renderCapacity: Int = 8192
    let captureQueue = PCMByteQueue(maxBytes: 14_400)
    let captureScratch = UnsafeMutablePointer<Int16>.allocate(capacity: 8192)
    let playbackScratch = UnsafeMutablePointer<Int16>.allocate(capacity: 8192)

    init() {
        renderScratch = UnsafeMutablePointer<Float>.allocate(capacity: renderCapacity)
        renderScratch.initialize(repeating: 0.0, count: renderCapacity)
    }

    deinit {
        renderScratch.deallocate()
        captureScratch.deallocate()
        playbackScratch.deallocate()
    }


}

private let vpioInputCallback: AURenderCallback = { (
    inRefCon,
    ioActionFlags,
    inTimeStamp,
    inBusNumber,
    inNumberFrames,
    _
) -> OSStatus in
    let ctx = Unmanaged<VPIOContext>.fromOpaque(inRefCon).takeUnretainedValue()
    let frames = Int(inNumberFrames)
    guard frames <= ctx.renderCapacity else { return kAudioUnitErr_TooManyFramesToProcess }

    var bufList = AudioBufferList(
        mNumberBuffers: 1,
        mBuffers: AudioBuffer(
            mNumberChannels: 1,
            mDataByteSize: UInt32(frames * MemoryLayout<Float>.size),
            mData: ctx.renderScratch
        )
    )
    let status = AudioUnitRender(
        ctx.audioUnit,
        ioActionFlags,
        inTimeStamp,
        inBusNumber,
        inNumberFrames,
        &bufList
    )
    if status != noErr {
        return status
    }

    for i in 0..<frames {
        let clamped = max(-1.0, min(1.0, ctx.renderScratch[i]))
        ctx.captureScratch[i] = Int16(clamped * 32767.0)
    }
    ctx.captureQueue.push(UnsafeRawBufferPointer(start: ctx.captureScratch, count: frames * 2))
    return noErr
}

private let vpioRenderCallback: AURenderCallback = { (
    inRefCon,
    _,
    _,
    _,
    inNumberFrames,
    ioData
) -> OSStatus in
    guard let ioData = ioData else { return noErr }
    let ctx = Unmanaged<VPIOContext>.fromOpaque(inRefCon).takeUnretainedValue()
    let abl = UnsafeMutableAudioBufferListPointer(ioData)
    let frames = Int(inNumberFrames)
    if abl.count > 0, let mData = abl[0].mData {
        let dst = mData.assumingMemoryBound(to: Float.self)
        guard frames <= ctx.renderCapacity else { return kAudioUnitErr_TooManyFramesToProcess }
        ctx.playbackQueue.renderSpeaker(ctx.playbackScratch, bytes: frames * 2, sampleRate: 16_000)
        for i in 0..<frames { dst[i] = Float(ctx.playbackScratch[i]) / 32768.0 }
    }
    return noErr
}

func runVPIODuplex() -> Int32 {
    var desc = AudioComponentDescription(
        componentType: kAudioUnitType_Output,
        componentSubType: kAudioUnitSubType_VoiceProcessingIO,
        componentManufacturer: kAudioUnitManufacturer_Apple,
        componentFlags: 0,
        componentFlagsMask: 0
    )
    guard let comp = AudioComponentFindNext(nil, &desc) else {
        fputs("VoiceProcessingIO component unavailable\n", stderr)
        return 2
    }

    var unitOpt: AudioUnit?
    var st = AudioComponentInstanceNew(comp, &unitOpt)
    guard st == noErr, let au = unitOpt else {
        fputs("AudioComponentInstanceNew failed: \(st)\n", stderr)
        return 3
    }

    let ctx = VPIOContext()
    ctx.audioUnit = au

    var one: UInt32 = 1
    st = AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &one, 4)
    if st != noErr { fputs("EnableIO input failed: \(st)\n", stderr); return 4 }

    st = AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &one, 4)
    if st != noErr { fputs("EnableIO output failed: \(st)\n", stderr); return 5 }

    var asbd = AudioStreamBasicDescription(
        mSampleRate: 16000.0,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4,
        mChannelsPerFrame: 1,
        mBitsPerChannel: 32,
        mReserved: 0
    )
    let asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    st = AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &asbd, asbdSize)
    if st != noErr { fputs("Set input format failed: \(st)\n", stderr); return 6 }

    st = AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &asbd, asbdSize)
    if st != noErr { fputs("Set output format failed: \(st)\n", stderr); return 7 }

    if #available(macOS 14.0, *) {
        var duck = AUVoiceIOOtherAudioDuckingConfiguration(mEnableAdvancedDucking: false, mDuckingLevel: .min)
        let _ = AudioUnitSetProperty(
            au,
            kAUVoiceIOProperty_OtherAudioDuckingConfiguration,
            kAudioUnitScope_Global,
            0,
            &duck,
            UInt32(MemoryLayout<AUVoiceIOOtherAudioDuckingConfiguration>.size)
        )
    }

    let refCon = Unmanaged.passUnretained(ctx).toOpaque()
    var inCb = AURenderCallbackStruct(inputProc: vpioInputCallback, inputProcRefCon: refCon)
    st = AudioUnitSetProperty(
        au,
        kAudioOutputUnitProperty_SetInputCallback,
        kAudioUnitScope_Global,
        1,
        &inCb,
        UInt32(MemoryLayout<AURenderCallbackStruct>.size)
    )
    if st != noErr { fputs("SetInputCallback failed: \(st)\n", stderr); return 8 }

    var outCb = AURenderCallbackStruct(inputProc: vpioRenderCallback, inputProcRefCon: refCon)
    st = AudioUnitSetProperty(
        au,
        kAudioUnitProperty_SetRenderCallback,
        kAudioUnitScope_Input,
        0,
        &outCb,
        UInt32(MemoryLayout<AURenderCallbackStruct>.size)
    )
    if st != noErr { fputs("SetRenderCallback failed: \(st)\n", stderr); return 9 }

    st = AudioUnitInitialize(au)
    if st != noErr { fputs("AudioUnitInitialize failed: \(st)\n", stderr); return 10 }

    startStdinMonitor(playbackQueue: ctx.playbackQueue)
    startCaptureWriter(ctx.captureQueue)

    st = AudioOutputUnitStart(au)
    if st != noErr {
        fputs("AudioOutputUnitStart failed: \(st)\n", stderr)
        AudioUnitUninitialize(au)
        AudioComponentInstanceDispose(au)
        return 11
    }

    while !gShouldStop {
        Thread.sleep(forTimeInterval: 0.05)
    }

    AudioOutputUnitStop(au)
    ctx.captureQueue.reportDrops("vpio capture")
    ctx.playbackQueue.reportDrops("vpio playback", sampleRate: 16_000)
    AudioUnitUninitialize(au)
    AudioComponentInstanceDispose(au)
    return 0
}

// MARK: - Mode 2: CoreAudio Process Tap (Inbound Meeting Audio Capture)

private func translatePIDToAudioProcessObject(_ pid: pid_t) -> AudioObjectID? {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var pidVal = pid
    var procObj: AudioObjectID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let st = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        UInt32(MemoryLayout<pid_t>.size),
        &pidVal,
        &size,
        &procObj
    )
    if st == noErr && procObj != kAudioObjectUnknown {
        return procObj
    }
    return nil
}

final class TapResampler {
    let targetRate: Double = 16000.0
    var sourceRate: Double
    var channels: Int
    var isNonInterleaved: Bool
    private var fracPos: Double = 0.0
    private var prevSample: Float = 0.0
    private var hasPrev = false
    let captureQueue = PCMByteQueue(maxBytes: 14_400)

    init(asbd: AudioStreamBasicDescription) {
        self.sourceRate = max(8000.0, asbd.mSampleRate)
        self.channels = max(1, Int(asbd.mChannelsPerFrame))
        self.isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
    }

    func process(bufferList: UnsafePointer<AudioBufferList>) {
        let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: bufferList))
        guard abl.count > 0 else { return }

        var mono = [Float]()
        if isNonInterleaved {
            let frames = Int(abl[0].mDataByteSize) / MemoryLayout<Float>.size
            if frames <= 0 { return }
            mono = [Float](repeating: 0.0, count: frames)
            let chCount = min(channels, abl.count)
            for c in 0..<chCount {
                guard let ptr = abl[c].mData?.assumingMemoryBound(to: Float.self) else { continue }
                for i in 0..<frames {
                    mono[i] += ptr[i]
                }
            }
            let scale = 1.0 / Float(max(1, chCount))
            for i in 0..<frames { mono[i] *= scale }
        } else {
            guard let ptr = abl[0].mData?.assumingMemoryBound(to: Float.self) else { return }
            let totalFloats = Int(abl[0].mDataByteSize) / MemoryLayout<Float>.size
            let frames = totalFloats / channels
            if frames <= 0 { return }
            mono = [Float](repeating: 0.0, count: frames)
            let scale = 1.0 / Float(channels)
            for i in 0..<frames {
                var sum: Float = 0.0
                let base = i * channels
                for c in 0..<channels {
                    sum += ptr[base + c]
                }
                mono[i] = sum * scale
            }
        }

        // Linear resample sourceRate -> 16000.0 Hz
        let step = sourceRate / targetRate
        var outSamples = [Int16]()
        outSamples.reserveCapacity(Int(Double(mono.count) / step) + 4)

        if !hasPrev && !mono.isEmpty {
            prevSample = mono[0]
            hasPrev = true
        }

        while fracPos < Double(mono.count) {
            let idx = Int(fracPos)
            let frac = Float(fracPos - Double(idx))
            let s0 = idx == 0 ? prevSample : mono[idx - 1]
            let s1 = mono[idx]
            let interp = s0 + (s1 - s0) * frac
            let clamped = max(-1.0, min(1.0, interp))
            outSamples.append(Int16(clamped * 32767.0))
            fracPos += step
        }
        if let last = mono.last {
            prevSample = last
        }
        fracPos -= Double(mono.count)

        if !outSamples.isEmpty {
            outSamples.withUnsafeBytes { captureQueue.push($0) }
        }
    }
}

func runSystemTap(excludePIDs: [pid_t]) -> Int32 {
    guard #available(macOS 14.2, *) else {
        fputs("AudioHardwareCreateProcessTap requires macOS 14.2+\n", stderr)
        return 20
    }

    var allPIDs = excludePIDs
    allPIDs.append(getpid())
    let excludedObjects = allPIDs.compactMap { translatePIDToAudioProcessObject($0) }

    let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: excludedObjects)
    tapDesc.uuid = UUID()
    tapDesc.muteBehavior = .unmuted

    var tapID: AudioObjectID = kAudioObjectUnknown
    var st = AudioHardwareCreateProcessTap(tapDesc, &tapID)
    guard st == noErr else {
        fputs("AudioHardwareCreateProcessTap failed: \(st)\n", stderr)
        return 21
    }
    defer { AudioHardwareDestroyProcessTap(tapID) }

    let aggDesc: [String: Any] = [
        kAudioAggregateDeviceNameKey: "VoiceCloneSystemTapAgg",
        kAudioAggregateDeviceUIDKey: "com.voiceclone.systemtap.\(UUID().uuidString)",
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceTapListKey: [
            [
                kAudioSubTapUIDKey: tapDesc.uuid.uuidString,
                kAudioSubTapDriftCompensationKey: true,
            ]
        ],
    ]
    var aggID: AudioObjectID = kAudioObjectUnknown
    st = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
    guard st == noErr else {
        fputs("AudioHardwareCreateAggregateDevice failed: \(st)\n", stderr)
        return 22
    }
    defer { AudioHardwareDestroyAggregateDevice(aggID) }

    var propAddr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamFormat,
        mScope: kAudioDevicePropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain
    )
    var asbd = AudioStreamBasicDescription()
    var propSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    st = AudioObjectGetPropertyData(aggID, &propAddr, 0, nil, &propSize, &asbd)
    guard st == noErr else {
        fputs("Failed to query tap stream format: \(st)\n", stderr)
        return 23
    }

    let resampler = TapResampler(asbd: asbd)
    var procID: AudioDeviceIOProcID?
    st = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { _, inInputData, _, _, _ in
        resampler.process(bufferList: inInputData)
    }
    guard st == noErr, let validProc = procID else {
        fputs("AudioDeviceCreateIOProcIDWithBlock failed: \(st)\n", stderr)
        return 24
    }
    defer { AudioDeviceDestroyIOProcID(aggID, validProc) }

    startStdinMonitor(playbackQueue: nil)
    startCaptureWriter(resampler.captureQueue)

    st = AudioDeviceStart(aggID, validProc)
    guard st == noErr else {
        fputs("AudioDeviceStart failed: \(st)\n", stderr)
        return 25
    }

    while !gShouldStop {
        Thread.sleep(forTimeInterval: 0.05)
    }
    AudioDeviceStop(aggID, validProc)
    resampler.captureQueue.reportDrops("tap capture")
    return 0
}

// Persistent default speaker with Core Audio's native AudioQueue callback.
private func runSpeaker() -> Int32 {
    // Streaming synthesis can arrive faster than real time. Allow a bounded
    // burst without evicting the oldest, about-to-be-played speech.
    let ring = PCMByteQueue(maxBytes: 576_000)
    var format = AudioStreamBasicDescription(
        mSampleRate: 24_000, mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
        mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2,
        mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0
    )
    let callback: AudioQueueOutputCallback = { state, queue, buffer in
        let pcm = Unmanaged<PCMByteQueue>.fromOpaque(state!).takeUnretainedValue()
        pcm.renderSpeaker(buffer.pointee.mAudioData, bytes: 960)
        buffer.pointee.mAudioDataByteSize = 960
        AudioQueueEnqueueBuffer(queue, buffer, 0, nil)
    }
    var handle: AudioQueueRef?
    let code = AudioQueueNewOutput(&format, callback, Unmanaged.passUnretained(ring).toOpaque(),
                                   nil, nil, 0, &handle)
    guard code == noErr, let queue = handle else { return 31 }
    defer {
        AudioQueueDispose(queue, true)
        ring.reportDrops("AudioQueue speaker")
    }
    for _ in 0..<3 {
        var buffer: AudioQueueBufferRef?
        guard AudioQueueAllocateBuffer(queue, 960, &buffer) == noErr, let ref = buffer else { return 32 }
        memset(ref.pointee.mAudioData, 0, 960)
        ref.pointee.mAudioDataByteSize = 960
        guard AudioQueueEnqueueBuffer(queue, ref, 0, nil) == noErr else { return 33 }
    }
    guard AudioQueueStart(queue, nil) == noErr else { return 34 }
    var ready: UInt8 = 82
    withUnsafeBytes(of: &ready) { writeStdoutFrame($0) }

    func readAll(_ ptr: UnsafeMutableRawPointer, _ count: Int) -> Bool {
        var offset = 0
        while offset < count && !gShouldStop {
            let n = Darwin.read(STDIN_FILENO, ptr + offset, count - offset)
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return false }
            offset += n
        }
        return offset == count
    }
    var buffer = [UInt8](repeating: 0, count: 192_000)
    var length: UInt32 = 0
    while !gShouldStop && withUnsafeMutableBytes(of: &length, { readAll($0.baseAddress!, $0.count) }) {
        let n = Int(UInt32(littleEndian: length))
        if n == 0 { ring.endSpeakerTurn(); continue }
        guard n <= buffer.count && n % 2 == 0,
              buffer.withUnsafeMutableBytes({ readAll($0.baseAddress!, n) }) else { return 35 }
        buffer.withUnsafeBytes { ring.push(UnsafeRawBufferPointer(rebasing: $0[..<n])) }
    }
    return 0
}

private func audioSelfTest() -> Int32 {
    let ring = PCMByteQueue(maxBytes: 192_000)
    let frame = [UInt8](repeating: 1, count: 960)
    var output = [UInt8](repeating: 0, count: frame.count)
    func write(_ data: [UInt8]) { data.withUnsafeBytes { ring.push($0) } }
    func render() -> [UInt8] {
        output.withUnsafeMutableBytes { ring.renderSpeaker($0.baseAddress!, bytes: $0.count) }
        return output
    }
    write([1])
    write(Array(frame.dropFirst()))
    if render().contains(1) { return 41 }
    for _ in 0..<8 { write(frame) }
    for _ in 0..<9 { if render() != frame { return 42 } }
    for _ in 0..<5 { write(frame) }
    ring.endSpeakerTurn()
    for _ in 0..<5 { if render() != frame { return 43 } }
    if render().contains(1) { return 44 }
    let capture = PCMByteQueue(maxBytes: 8)
    [UInt8](0..<12).withUnsafeBytes { capture.push($0) }
    guard output.withUnsafeMutableBytes({ capture.popFrame(into: $0.baseAddress!, bytes: 8) }),
          Array(output.prefix(8)) == Array(4..<12) else { return 45 }
    return 0
}

// MARK: - CLI Entrypoint

signal(SIGINT, handleSignal)
signal(SIGTERM, handleSignal)
signal(SIGPIPE, SIG_IGN)

var mode = ""
var excludePIDs: [pid_t] = []
var idx = 1
let args = CommandLine.arguments
while idx < args.count {
    switch args[idx] {
    case "--mode":
        if idx + 1 < args.count { mode = args[idx + 1]; idx += 1 }
    case "--exclude-pid":
        if idx + 1 < args.count, let p = Int32(args[idx + 1]) {
            excludePIDs.append(p)
            idx += 1
        }
    default:
        break
    }
    idx += 1
}

switch mode {
case "vpio-duplex":
    exit(runVPIODuplex())
case "self-test":
    exit(audioSelfTest())
case "speaker":
    exit(runSpeaker())
case "system-tap":
    exit(runSystemTap(excludePIDs: excludePIDs))
default:
    fputs("Usage: VoiceCloneAudioEngine --mode <vpio-duplex|system-tap|speaker> [--exclude-pid PID]\n", stderr)
    exit(1)
}
