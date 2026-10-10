// Cargo runner for `wasm32-unknown-unknown` tests: instantiates the module
// with no imports (instantiating fails if it needs any), so with no clock, no
// filesystem and no source of randomness. The target cannot print, so
// libtest's output is lost; a failing test panics, which aborts, which traps.
import { readFile } from 'node:fs/promises';

const { instance } = await WebAssembly.instantiate(await readFile(process.argv[2]), {});
process.exitCode = instance.exports.main(0, 0);
