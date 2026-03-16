#!/usr/bin/env node
/**
 * WebRTC AEC3 runner: reads lpb (far-end) and mic (near-end) WAVs,
 * runs AEC3, writes echo-cancelled WAV. Used by evaluation server.
 * Usage: node run_aec3.js --lpb <path> --mic <path> --out <path>
 *
 * Requires: npm install in this directory (or build external/webrtcaec3.js and set path).
 * WAVs: 16 kHz mono, 16-bit PCM.
 */

import { readFileSync, writeFileSync } from 'fs';

const SAMPLE_RATE = 16000;
const AEC3_SAMPLE_RATE = 48000; // AEC3 lib expects 32k or 48k
const FRAME_SAMPLES = 160;      // 10 ms at 16 kHz

function parseArgs() {
  const args = {};
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--lpb' && process.argv[i + 1]) args.lpb = process.argv[++i];
    else if (process.argv[i] === '--mic' && process.argv[i + 1]) args.mic = process.argv[++i];
    else if (process.argv[i] === '--out' && process.argv[i + 1]) args.out = process.argv[++i];
  }
  if (!args.lpb || !args.mic || !args.out) {
    console.error('Usage: node run_aec3.js --lpb <path> --mic <path> --out <path>');
    process.exit(1);
  }
  return args;
}

/** Read mono WAV (16-bit PCM or 32-bit float), return float32 array (normalized -1..1). */
function readWav(path) {
  const buf = readFileSync(path);
  if (buf.length < 44) throw new Error(`WAV too small: ${path}`);
  const fmt = buf.readUInt16LE(20);
  const bits = buf.readUInt16LE(34);
  const dataStart = 44;
  const data = buf.subarray(dataStart);
  if (fmt === 3 && bits === 32) {
    const n = data.length / 4;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = data.readFloatLE(i * 4);
    return out;
  }
  if (fmt === 1 && bits === 16) {
    const n = data.length / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = data.readInt16LE(i * 2) / 32768;
    return out;
  }
  throw new Error(`Unsupported WAV format: fmt=${fmt} bits=${bits}`);
}

/** Write float32 array as 16-bit mono WAV at 16 kHz. */
function writeWav(path, samples) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + samples.length * 2, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);   // PCM
  header.writeUInt16LE(1, 22);   // mono
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(samples.length * 2, 40);
  const body = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
    body.writeInt16LE(s, i * 2);
  }
  writeFileSync(path, Buffer.concat([header, body]));
}

async function loadAec3() {
  try {
    const mod = await import('@ennuicastr/webrtcaec3.js');
    return await mod.default();
  } catch (e) {
    // Fallback: try local build from external (file URL for Node ESM)
    const path = new URL('../../external/webrtcaec3.js/dist/webrtcaec3-0.3.0.js', import.meta.url);
    try {
      const mod = await import(path.href);
      return await mod.default();
    } catch (e2) {
      console.error('Failed to load AEC3. Run: npm install in evaluation/webrtc-aec3-node');
      console.error(e.message);
      process.exit(1);
    }
  }
}

async function main() {
  const { lpb, mic, out } = parseArgs();
  const lpbSamples = readWav(lpb);
  const micSamples = readWav(mic);
  const n = Math.min(lpbSamples.length, micSamples.length);
  if (n === 0) {
    console.error('Empty or missing audio');
    process.exit(1);
  }

  const WebRtcAec3 = await loadAec3();
  const aec3 = new WebRtcAec3.AEC3(AEC3_SAMPLE_RATE, 1, 1);
  const opts = { sampleRateIn: SAMPLE_RATE, sampleRateOut: SAMPLE_RATE };

  const outChunks = [];
  let offset = 0;
  while (offset < n) {
    const len = Math.min(FRAME_SAMPLES, n - offset);
    const lpbFrame = [lpbSamples.slice(offset, offset + len)];
    const micFrame = [micSamples.slice(offset, offset + len)];
    aec3.analyze(lpbFrame, opts);
    const outLen = aec3.processSize(micFrame, opts);
    const outFrame = [new Float32Array(outLen)];
    aec3.process(outFrame, micFrame, opts);
    outChunks.push(outFrame[0]);
    offset += len;
  }
  aec3.free();

  const totalOut = outChunks.reduce((a, b) => a + b.length, 0);
  const combined = new Float32Array(totalOut);
  let pos = 0;
  for (const chunk of outChunks) {
    combined.set(chunk, pos);
    pos += chunk.length;
  }
  // If resampling changed length, take first n samples to match input length
  const toWrite = totalOut >= n ? combined.subarray(0, n) : combined;
  writeWav(out, toWrite);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
