# WebRTC AEC3 runner

Runs [ennuicastr/webrtcaec3.js](https://github.com/ennuicastr/webrtcaec3.js) for the evaluation server. The server calls this script as a subprocess with `--lpb`, `--mic`, and `--out` WAV paths (16 kHz mono).

## Setup

```bash
npm install
```

This installs `@ennuicastr/webrtcaec3.js` from npm. If the package is unavailable, you can build from the cloned repo:

1. Clone and build [ennuicastr/webrtcaec3.js](https://github.com/ennuicastr/webrtcaec3.js) into `external/webrtcaec3.js` (requires Emscripten: `emcc`).
2. Run `make` in that directory to produce `dist/webrtcaec3-0.3.0.js`.
3. The runner will try to load from `external/webrtcaec3.js/dist/` if the npm package is not installed.

## Usage (called by server)

```bash
node run_aec3.js --lpb /path/to/lpb.wav --mic /path/to/mic.wav --out /path/to/out.wav
```

Input WAVs: 16 kHz mono (16-bit PCM or 32-bit float). Output: 16 kHz mono 16-bit PCM WAV.
