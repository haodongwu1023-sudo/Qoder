class Pcm16DownsampleProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._stash = [];
    this._targetRate = 16000;
    this._frameSamples = 640; // 40ms @ 16kHz
  }

  process(inputs) {
    const input = inputs && inputs[0] && inputs[0][0];
    if (!input || !input.length) return true;

    let sq = 0;
    for (let i = 0; i < input.length; i++) {
      sq += input[i] * input[i];
    }
    const rms = Math.sqrt(sq / input.length);

    if (sampleRate === this._targetRate) {
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        this._stash.push(s < 0 ? s * 0x8000 : s * 0x7fff);
      }
    } else {
      const ratio = sampleRate / this._targetRate;
      const outLen = Math.floor(input.length / ratio);
      for (let i = 0; i < outLen; i++) {
        const idx = i * ratio;
        const lo = Math.floor(idx);
        const hi = Math.min(input.length - 1, lo + 1);
        const frac = idx - lo;
        const val = input[lo] * (1 - frac) + input[hi] * frac;
        const s = Math.max(-1, Math.min(1, val));
        this._stash.push(s < 0 ? s * 0x8000 : s * 0x7fff);
      }
    }

    while (this._stash.length >= this._frameSamples) {
      const frame = new Int16Array(this._stash.splice(0, this._frameSamples));
      this.port.postMessage({ pcm: frame.buffer, rms }, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm16-downsample-processor', Pcm16DownsampleProcessor);
