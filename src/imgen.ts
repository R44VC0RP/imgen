#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { emitKeypressEvents, type Key } from 'node:readline';
import { parseArgs } from 'node:util';
import { setTimeout as sleep } from 'node:timers/promises';
import OpenAI, { toFile } from 'openai';
import type { ImageGenerateParamsBase, ImagesResponse } from 'openai/resources/images';
import sharp from 'sharp';

type ImageRequest = ImageGenerateParamsBase & {
  model: string;
  size: string;
  quality: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | 'auto';
  background: 'transparent' | 'opaque' | 'auto';
  output_format: 'png' | 'webp' | 'jpeg';
  n: number;
  input_fidelity?: 'low' | 'high';
};

interface CharacterReference {
  name: string;
  handle: string;
  description?: string;
  images: string[];
}

interface ResponseRequest {
  prompt: string;
  model: string;
  image_model: string;
  action: 'generate' | 'edit' | 'auto';
  size: string;
  quality: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | 'auto';
  background: 'transparent' | 'opaque' | 'auto';
  output_format: 'png' | 'webp' | 'jpeg';
  output_compression?: number;
  moderation?: 'auto' | 'low';
  previous_response_id?: string;
  characters: CharacterReference[];
}

interface SavedImage {
  path: string;
  width: number;
  height: number;
  format: string;
  bytes: number;
  has_alpha: boolean;
  has_transparency: boolean;
  fully_transparent_fraction: number;
}

interface Job {
  id: string;
  kind: 'generate' | 'edit' | 'respond';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';
  created_at: string;
  updated_at: string;
  started_at?: string;
  finished_at?: string;
  elapsed_seconds?: number;
  heartbeat_stale?: boolean;
  pid?: number;
  request: ImageRequest | ResponseRequest;
  response_id?: string;
  request_id?: string | null;
  timeout_ms: number;
  output_paths: string[];
  images: string[];
  mask?: string;
  outputs: SavedImage[];
  previews: SavedImage[];
  warnings: string[];
  last_event?: string;
  usage?: ImagesResponse['usage'] | unknown;
  actual?: Partial<Pick<ImageRequest, 'size' | 'quality' | 'background' | 'output_format'>>;
  error?: { message: string; status?: number; code?: string | null };
}

const home = process.env.IMGEN_HOME;
const configDir = home || path.join(process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config'), 'imgen');
const stateDir = home || path.join(process.env.XDG_STATE_HOME || path.join(os.homedir(), '.local/state'), 'imgen');
const jobsDir = path.join(stateDir, 'jobs');
const credentialsFile = path.join(configDir, 'credentials.json');
const charactersDir = path.join(configDir, 'characters');
const activeStates = new Set(['queued', 'running']);
const now = () => new Date().toISOString();
const boolean = { type: 'boolean' } as const;
const string = { type: 'string' } as const;
const common = { help: { ...boolean, short: 'h' }, json: boolean };
const imageOptions = {
  ...common, prompt: { ...string, short: 'p' }, 'prompt-file': string,
  out: { ...string, short: 'o' }, model: string, size: string, quality: string,
  background: string, 'output-format': string, 'output-compression': string,
  n: string, moderation: string, user: string, stream: boolean,
  'partial-images': string, timeout: string, wait: boolean,
};
const options = {
  generate: imageOptions,
  edit: { ...imageOptions, image: { ...string, multiple: true as const }, mask: string, 'input-fidelity': string },
  respond: {
    ...common, prompt: { ...string, short: 'p' }, 'prompt-file': string, out: { ...string, short: 'o' },
    model: string, 'image-model': string, action: string, size: string, quality: string,
    background: string, 'output-format': string, 'output-compression': string, moderation: string,
    'previous-response': string, timeout: string, wait: boolean,
  },
  status: { ...common, watch: boolean, limit: string, state: string },
  login: { ...common, stdin: boolean },
  character: { ...common, image: { ...string, multiple: true as const }, description: string },
};

type AllOptions = typeof options.edit & typeof options.respond & typeof options.status & typeof options.login & typeof options.character;
type Values = ReturnType<typeof parseArgs<{ options: AllOptions }>>['values'];

const help = `imgen - OpenAI image generation with persistent background jobs

Usage:
  imgen login [--stdin] [--json]
  imgen generate [PROMPT] --out FILE [OPTIONS]
  imgen edit [PROMPT] --image FILE [--image FILE ...] --out FILE [OPTIONS]
  imgen respond [PROMPT] --out FILE [OPTIONS]
  imgen character add NAME --image FILE [--image FILE ...] [--description TEXT]
  imgen character list|show NAME|remove NAME [--json]
  imgen status [JOB_ID] [--watch] [--limit N] [--state STATE] [--json]

Image options:
  -p, --prompt TEXT            Prompt (or use the positional prompt)
  --prompt-file FILE           Read prompt from a file; '-' reads stdin
  -o, --out FILE               Output path; never overwrites existing files
  --model MODEL               Default: gpt-image-2.5-sunburst
  --size WIDTHxHEIGHT|auto     Default: 1024x1024; custom GPT Image 2/2.5 sizes
  --quality QUALITY           low|medium|high|xhigh|max|auto; default: high
                              xhigh and max require GPT Image 2.5
  --background transparent|opaque|auto  Default: transparent (JPEG: auto)
  --output-format png|webp|jpeg  Default: inferred from --out, otherwise png
  --output-compression 0..100  JPEG/WebP only; API default when omitted
  --n 1..10                   Images per job; FILE becomes FILE-1, FILE-2, ...
  --moderation auto|low        API default when omitted
  --user ID                    End-user identifier for OpenAI abuse monitoring
  --stream                    Receive streaming events and save any previews
  --partial-images 0..3        Request streaming previews (additional API cost)
  --timeout SECONDS           Total API request deadline; default: 900
  --wait                      Wait for the background job; fail if it fails
  --json                      Machine-readable result, no base64 payloads

Edit-only options:
  --image FILE                Repeat for up to 16 PNG/JPEG/WebP references
  --mask FILE                 PNG alpha mask, same dimensions as first image
  --input-fidelity low|high    Older GPT Image models only; omit for Image 2/2.5

Responses API options:
  --model MODEL                Reasoning model; default: gpt-5.6
  --image-model MODEL          Image tool model; default: gpt-image-2.5-sunburst
  --action auto|generate|edit  Default: auto
  --previous-response ID       Continue a prior Responses API image conversation
  Prompts may mention saved characters as @Handle; their reference images are attached.

Characters:
  character add copies private reference images into ~/.config/imgen/characters.
  Handles use letters, numbers, underscores, and hyphens and are case-insensitive.

Status:
  No ID lists the 20 newest jobs; --limit N changes this.
  --state queued|running|completed|failed|interrupted filters the list.
  --watch polls each second; with an ID it stops when that job finishes.
  --watch --json emits one JSON snapshot per line (NDJSON).
  Exit codes: 0 success, 1 error/failed job, 2 usage error.

Authentication:
  imgen login prompts for an OpenAI API key without displaying it.
  --stdin reads the key from stdin for secret-manager integration.
  OPENAI_API_KEY overrides the saved key. Credentials are stored with mode 600.
  OPENAI_ORG_ID and OPENAI_PROJECT_ID are supported.
  IMGEN_HOME overrides both credential and job storage for isolated use.

Notes:
  Jobs run locally and survive terminal exit, but not a reboot or forced kill.
  No automatic request retries: avoids accidentally paying for duplicate images.
  --wait interruption does not cancel the background job.
  GPT Image 2/2.5 sizes: edges divisible by 16, max edge 3840, max ratio 3:1,
  total pixels 655360..8294400. Above 3686400 pixels is experimental.
  Transparency is in preview. Prompts can override it; results check actual alpha.
  Output formats are PNG/JPEG/WebP, not SVG. GPT Image always returns base64.

Examples:
  imgen generate "An isolated ceramic flower, no backdrop" -o flower.png
  imgen generate --prompt-file brief.txt -o hero.webp --size 1536x1024 --json
  imgen edit "Make the petals blue" --image flower.png -o blue.png --wait
  imgen character add Alan --image alan-front.png --image alan-side.png
  imgen respond "@Alan and @Ryan sitting in a boat" -o boat.png --wait
  imgen status JOB_ID --watch --json`;

function usage(message: string): never {
  throw Object.assign(new Error(message), { exitCode: 2 });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function privateDir(dir: string) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
}

function atomicJSON(file: string, value: unknown) {
  const temp = `${file}.${randomUUID()}.tmp`;
  try {
    fs.writeFileSync(temp, JSON.stringify(value, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
    fs.renameSync(temp, file);
  } finally {
    fs.rmSync(temp, { force: true });
  }
}

function characterHandle(value: string): string {
  if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(value)) usage('Character names must start with a letter and use only letters, numbers, underscores, or hyphens.');
  return value.toLowerCase();
}

function characterFile(handle: string): string {
  return path.join(charactersDir, handle, 'character.json');
}

function readCharacter(name: string): CharacterReference {
  const handle = characterHandle(name);
  try {
    const character = JSON.parse(fs.readFileSync(characterFile(handle), 'utf8')) as CharacterReference;
    if (character.handle !== handle || !Array.isArray(character.images)) throw new Error('invalid metadata');
    return character;
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') throw new Error(`Unknown character: @${name}`);
    throw new Error(`Cannot read character @${name}: ${errorMessage(error)}`);
  }
}

async function character(values: Values, positionals: string[]) {
  const [action, name, ...extra] = positionals;
  if (extra.length || !action || !['add', 'list', 'show', 'remove'].includes(action)) usage('Use: imgen character add NAME, list, show NAME, or remove NAME.');
  if (action === 'list') {
    if (name || values.image?.length || values.description) usage('character list takes no name or image options.');
    let characters: CharacterReference[] = [];
    try {
      characters = fs.readdirSync(charactersDir, { withFileTypes: true }).filter(entry => entry.isDirectory()).map(entry => readCharacter(entry.name)).sort((a, b) => a.name.localeCompare(b.name));
    } catch (error) {
      if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) throw error;
    }
    if (values.json) console.log(JSON.stringify({ characters }));
    else if (!characters.length) console.log('No saved characters.');
    else for (const item of characters) console.log(`@${item.name}  ${item.images.length} image${item.images.length === 1 ? '' : 's'}${item.description ? `  ${item.description}` : ''}`);
    return;
  }
  if (!name) usage(`character ${action} requires a name.`);
  const handle = characterHandle(name);
  if (action === 'show') {
    const item = readCharacter(handle);
    if (values.json) console.log(JSON.stringify(item));
    else console.log(`@${item.name}\nHandle: @${item.handle}\nImages:\n${item.images.map(file => `  ${file}`).join('\n')}${item.description ? `\nDescription: ${item.description}` : ''}`);
    return;
  }
  if (action === 'remove') {
    if (values.image?.length || values.description) usage('character remove does not accept image options.');
    readCharacter(handle);
    fs.rmSync(path.dirname(characterFile(handle)), { recursive: true });
    if (values.json) console.log(JSON.stringify({ removed: true, handle }));
    else console.log(`Removed @${name}.`);
    return;
  }
  const sources = values.image ?? [];
  if (!sources.length) usage('character add requires at least one --image FILE.');
  const dir = path.dirname(characterFile(handle));
  if (fs.existsSync(dir)) usage(`Character already exists: @${name}. Remove it first to replace its references.`);
  privateDir(charactersDir);
  privateDir(dir);
  try {
    const images: string[] = [];
    for (const [index, source] of sources.entries()) {
      const file = path.resolve(source);
      const stat = fs.statSync(file);
      if (!stat.isFile() || stat.size >= 50 * 1024 * 1024) usage(`Reference must be a file smaller than 50 MB: ${file}`);
      const meta = await sharp(file).metadata();
      if (!['png', 'jpeg', 'webp'].includes(meta.format)) usage(`Reference must be PNG, JPEG, or WebP: ${file}`);
      const extension = meta.format === 'jpeg' ? '.jpg' : `.${meta.format}`;
      const destination = path.join(dir, `reference-${index + 1}${extension}`);
      fs.copyFileSync(file, destination, fs.constants.COPYFILE_EXCL);
      fs.chmodSync(destination, 0o600);
      images.push(destination);
    }
    const item: CharacterReference = { name, handle, ...(values.description ? { description: values.description } : {}), images };
    atomicJSON(characterFile(handle), item);
    if (values.json) console.log(JSON.stringify(item));
    else console.log(`Saved @${name} with ${images.length} reference image${images.length === 1 ? '' : 's'}.`);
  } catch (error) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw error;
  }
}

function getKey(): string {
  if (process.env.OPENAI_API_KEY?.trim()) return process.env.OPENAI_API_KEY.trim();
  try {
    const key = JSON.parse(fs.readFileSync(credentialsFile, 'utf8')).api_key;
    if (typeof key === 'string' && key.trim()) return key.trim();
  } catch (error) {
    if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) throw new Error('Cannot read saved credentials. Run imgen login again.');
  }
  throw new Error('No OpenAI API key. Run imgen login or set OPENAI_API_KEY.');
}

async function login(values: Values) {
  let key: string;
  if (values.stdin) {
    if (process.stdin.isTTY) usage('--stdin expects a piped key. Use imgen login for the hidden prompt.');
    key = fs.readFileSync(0, 'utf8').trim();
  } else {
    if (!process.stdin.isTTY || !process.stderr.isTTY) usage('Login needs a terminal. Pipe a key with imgen login --stdin instead.');
    key = await new Promise<string>((resolve, reject) => {
      let value = '';
      emitKeypressEvents(process.stdin);
      process.stdin.setRawMode(true);
      process.stdin.resume();
      const finish = (error?: Error) => {
        process.stdin.off('keypress', onKey);
        process.stdin.setRawMode(false);
        process.stdin.pause();
        process.stderr.write('\n');
        error ? reject(error) : resolve(value.trim());
      };
      const onKey = (text: string | undefined, event: Key = {}) => {
        if (event.ctrl && (event.name === 'c' || event.name === 'd')) return finish(new Error('Login cancelled.'));
        if (event.name === 'return' || event.name === 'enter') return finish();
        if (event.name === 'backspace') value = value.slice(0, -1);
        else if (!event.ctrl && !event.meta && text && !/[\x00-\x1f\x7f]/.test(text)) value += text;
      };
      process.stdin.on('keypress', onKey);
      process.stderr.write('OpenAI API key (input hidden): ');
    });
  }
  if (!key || /\s/.test(key)) usage('API key must be a nonempty value without whitespace.');
  privateDir(configDir);
  atomicJSON(credentialsFile, { api_key: key });
  const result = { saved: true, path: credentialsFile, verified: false, environment_override: Boolean(process.env.OPENAI_API_KEY?.trim()) };
  if (values.json) console.log(JSON.stringify(result));
  else {
    console.log(`API key saved to ${credentialsFile} (mode 600). No API request was made.`);
    if (result.environment_override) console.error('OPENAI_API_KEY is set and will take precedence over this saved key.');
  }
}

function integer(value: string, name: string, min: number, max: number): number {
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < min || Number(value) > max) {
    usage(`--${name} must be an integer from ${min} to ${max}.`);
  }
  return Number(value);
}

function choice<const T extends readonly string[]>(value: string, name: string, choices: T): T[number] {
  if (!choices.includes(value)) usage(`--${name} must be ${choices.join(', ')}.`);
  return value as T[number];
}

function jobFile(id: string): string {
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id)) usage('Invalid job ID. Use the full ID from imgen status.');
  return path.join(jobsDir, `${id}.json`);
}

function readJob(id: string): Job {
  let job: Job;
  try {
    job = JSON.parse(fs.readFileSync(jobFile(id), 'utf8'));
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') throw new Error(`Job not found: ${id}`);
    throw error;
  }
  if (activeStates.has(job.status)) {
    let alive = false;
    if (job.pid) {
      try { process.kill(job.pid, 0); alive = true; } catch (error) { alive = error instanceof Error && 'code' in error && error.code === 'EPERM'; }
    }
    const age = Date.now() - Date.parse(job.updated_at);
    // A sleeping Mac can delay heartbeats even while the worker is still alive.
    if (alive && age > 60000) job.heartbeat_stale = true;
    if (!alive && age > 15000) {
      job.status = 'interrupted';
      job.error = { message: 'The local worker stopped. OpenAI may still have processed the request; no automatic retry was made.' };
    }
  }
  job.elapsed_seconds = Math.max(0, Math.round(((job.finished_at ? Date.parse(job.finished_at) : Date.now()) - Date.parse(job.started_at || job.created_at)) / 1000));
  return job;
}

function printJob(job: Job, json?: boolean) {
  if (json) return console.log(JSON.stringify(job));
  console.log(`${job.id}  ${job.status}  ${job.kind}`);
  console.log(`Created: ${job.created_at}`);
  console.log(`Elapsed: ${job.elapsed_seconds}s  Images: ${job.outputs.length}/${job.kind === 'respond' ? 1 : (job.request as ImageRequest).n}`);
  console.log(`Model: ${job.request.model}  Size: ${job.request.size}  Quality: ${job.request.quality}`);
  if (job.response_id) console.log(`Response: ${job.response_id}`);
  if (job.request_id) console.log(`Request: ${job.request_id}`);
  if (job.usage) {
    const usage = job.usage as { input_tokens?: number; output_tokens?: number; total_tokens?: number };
    console.log(`Tokens: ${usage.input_tokens ?? '?'} input, ${usage.output_tokens ?? '?'} output, ${usage.total_tokens ?? '?'} total`);
  }
  if (job.heartbeat_stale) console.error('Warning: Worker is alive but its heartbeat is stale; it may be paused or waking from sleep.');
  for (const output of job.outputs) console.log(`Output: ${output.path} (${output.width}x${output.height}, transparency: ${output.has_transparency})`);
  for (const preview of job.previews) console.log(`Preview: ${preview.path}`);
  for (const warning of job.warnings) console.error(`Warning: ${warning}`);
  if (job.error) console.error(`Error: ${job.error.message}`);
}

async function status(values: Values, positionals: string[]) {
  if (positionals.length > 1) usage('status accepts at most one job ID.');
  const id = positionals[0];
  if (id && (values.limit || values.state)) usage('--limit and --state apply only to the job list.');
  const limit = integer(values.limit ?? '20', 'limit', 1, 10000);
  if (values.state) choice(values.state, 'state', ['queued', 'running', 'completed', 'failed', 'interrupted']);
  do {
    if (id) {
      const job = readJob(id);
      printJob(job, values.json);
      if (!activeStates.has(job.status)) {
        if (job.status !== 'completed') process.exitCode = 1;
        return;
      }
    } else {
      const files = fs.existsSync(jobsDir) ? fs.readdirSync(jobsDir).filter(file => file.endsWith('.json')) : [];
      const jobs = files.map(file => readJob(file.slice(0, -5)))
        .filter(job => !values.state || job.status === values.state)
        .sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, limit);
      if (values.json) console.log(JSON.stringify({ jobs }));
      else if (!jobs.length) console.log('No matching jobs.');
      else for (const job of jobs) console.log(`${job.id}  ${job.status.padEnd(11)}  ${job.elapsed_seconds}s  ${job.outputs.length}/${job.kind === 'respond' ? 1 : (job.request as ImageRequest).n} images  ${job.outputs[0]?.path || job.output_paths[0]}${job.heartbeat_stale ? '  (stale heartbeat)' : ''}${job.error ? `  ${job.error.message}` : ''}`);
    }
    if (values.watch) await sleep(1000);
  } while (values.watch);
}

function promptFrom(values: Values, positionals: string[]): string {
  if (positionals.length > 1) usage('Quote the positional prompt, or use --prompt-file.');
  const sources = [values.prompt, values['prompt-file'], positionals[0]].filter(value => value !== undefined);
  if (sources.length !== 1) usage('Supply exactly one prompt: positional text, --prompt, or --prompt-file.');
  if (values['prompt-file'] === '-' && process.stdin.isTTY) usage('--prompt-file - expects piped input.');
  const prompt = values['prompt-file'] !== undefined
    ? fs.readFileSync(values['prompt-file'] === '-' ? 0 : values['prompt-file'], 'utf8')
    : values.prompt ?? positionals[0];
  if (!prompt.trim() || [...prompt].length > 32000) usage('Prompt must contain 1 to 32000 characters.');
  return prompt;
}

async function submitResponse(values: Values, positionals: string[]) {
  const prompt = promptFrom(values, positionals);
  if (!values.out) usage('--out FILE is required.');
  const out = path.resolve(values.out);
  const extension = path.extname(out).toLowerCase();
  const formats: Record<string, ResponseRequest['output_format']> = { '.png': 'png', '.webp': 'webp', '.jpg': 'jpeg', '.jpeg': 'jpeg' };
  const format = choice(values['output-format'] ?? formats[extension] ?? 'png', 'output-format', ['png', 'webp', 'jpeg']);
  if (extension && formats[extension] !== format) usage('--out extension must match --output-format (png, webp, jpg, jpeg), or have no extension.');
  const background = choice(values.background ?? (format === 'jpeg' ? 'auto' : 'transparent'), 'background', ['transparent', 'opaque', 'auto']);
  if (format === 'jpeg' && background === 'transparent') usage('JPEG cannot have a transparent background. Use PNG/WebP or --background opaque.');
  const handles = [...new Set([...prompt.matchAll(/@([A-Za-z][A-Za-z0-9_-]*)/g)].map(match => match[1].toLowerCase()))];
  const characters = handles.map(readCharacter);
  const request: ResponseRequest = {
    prompt,
    model: values.model ?? 'gpt-5.6',
    image_model: values['image-model'] ?? 'gpt-image-2.5-sunburst',
    action: choice(values.action ?? 'auto', 'action', ['auto', 'generate', 'edit']),
    size: values.size ?? '1024x1024',
    quality: choice(values.quality ?? 'high', 'quality', ['low', 'medium', 'high', 'xhigh', 'max', 'auto']),
    background,
    output_format: format,
    characters,
  };
  if ((request.quality === 'xhigh' || request.quality === 'max') && !/^gpt-image-2\.5-(sunburst|flare)(-\d{4}-\d{2}-\d{2})?$/.test(request.image_model)) {
    usage('--quality xhigh and max require a GPT Image 2.5 Sunburst or Flare model.');
  }
  if (values['output-compression'] !== undefined) {
    if (format === 'png') usage('--output-compression is only supported for JPEG and WebP.');
    request.output_compression = integer(values['output-compression'], 'output-compression', 0, 100);
  }
  if (values.moderation !== undefined) request.moderation = choice(values.moderation, 'moderation', ['auto', 'low']);
  if (values['previous-response']) request.previous_response_id = values['previous-response'];
  const timeout = integer(values.timeout ?? '900', 'timeout', 1, 86400) * 1000;
  getKey();
  privateDir(jobsDir);
  const id = randomUUID();
  const assetsDir = path.join(jobsDir, id);
  privateDir(assetsDir);
  const copiedCharacters: CharacterReference[] = [];
  for (const item of characters) {
    const images: string[] = [];
    for (const [index, source] of item.images.entries()) {
      const destination = path.join(assetsDir, `${item.handle}-${index + 1}${path.extname(source)}`);
      fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
      fs.chmodSync(destination, 0o600);
      images.push(destination);
    }
    copiedCharacters.push({ ...item, images });
  }
  request.characters = copiedCharacters;
  fs.mkdirSync(path.dirname(out), { recursive: true });
  if (fs.existsSync(out)) usage(`Output already exists: ${out}. Choose a new --out path.`);
  fs.accessSync(path.dirname(out), fs.constants.W_OK);
  const job: Job = {
    id, kind: 'respond', status: 'queued', created_at: now(), updated_at: now(), request,
    timeout_ms: timeout, output_paths: [out], images: copiedCharacters.flatMap(item => item.images), outputs: [], previews: [], warnings: [],
  };
  atomicJSON(jobFile(id), job);
  try {
    const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '__worker', id], { detached: true, stdio: 'ignore' });
    await once(child, 'spawn');
    child.unref();
  } catch (error) {
    job.status = 'failed';
    job.error = { message: `Unable to start worker: ${errorMessage(error)}` };
    job.finished_at = job.updated_at = now();
    atomicJSON(jobFile(id), job);
    throw error;
  }
  if (!values.wait) {
    if (values.json) console.log(JSON.stringify({ id, status: 'queued', output_paths: [out] }));
    else console.log(`Started ${id}\nCheck: imgen status ${id} --watch`);
    return;
  }
  console.error(`Waiting for ${id}. Interrupting this command will not cancel it.`);
  while (true) {
    const current = readJob(id);
    if (!activeStates.has(current.status)) {
      printJob(current, values.json);
      if (current.status !== 'completed') process.exitCode = 1;
      return;
    }
    await sleep(500);
  }
}

async function submit(kind: Job['kind'], values: Values, positionals: string[]) {
  const prompt = promptFrom(values, positionals);
  if (!values.out) usage('--out FILE is required.');
  const out = path.resolve(values.out);
  const extension = path.extname(out).toLowerCase();
  const formats: Record<string, ImageRequest['output_format']> = { '.png': 'png', '.webp': 'webp', '.jpg': 'jpeg', '.jpeg': 'jpeg' };
  const inferredFormat = formats[extension];
  const format = choice(values['output-format'] ?? inferredFormat ?? 'png', 'output-format', ['png', 'webp', 'jpeg']);
  if (extension && inferredFormat !== format) usage('--out extension must match --output-format (png, webp, jpg, jpeg), or have no extension.');
  const request: ImageRequest = {
    model: values.model ?? 'gpt-image-2.5-sunburst', prompt,
    size: values.size ?? '1024x1024',
    quality: choice(values.quality ?? 'high', 'quality', ['low', 'medium', 'high', 'xhigh', 'max', 'auto']),
    background: choice(values.background ?? (format === 'jpeg' ? 'auto' : 'transparent'), 'background', ['transparent', 'opaque', 'auto']),
    output_format: format,
    n: integer(values.n ?? '1', 'n', 1, 10),
  };
  if (!/^(gpt-image-[\w.-]+|chatgpt-image-latest)$/.test(request.model)) usage('--model must be a GPT Image model (not DALL-E).');
  if ((request.quality === 'xhigh' || request.quality === 'max') && !/^gpt-image-2\.5-(sunburst|flare)(-\d{4}-\d{2}-\d{2})?$/.test(request.model)) {
    usage('--quality xhigh and max require a GPT Image 2.5 Sunburst or Flare model.');
  }
  if (format === 'jpeg' && request.background === 'transparent') usage('JPEG cannot have a transparent background. Use PNG/WebP or --background opaque.');
  const warnings: string[] = [];
  if (request.size !== 'auto') {
    if (!/^\d+x\d+$/.test(request.size)) usage('--size must be WIDTHxHEIGHT or auto.');
    const [width, height] = request.size.split('x').map(Number);
    if (request.model.startsWith('gpt-image-2')) {
      const pixels = width * height;
      if (width % 16 || height % 16 || Math.max(width, height) > 3840 || Math.max(width, height) / Math.min(width, height) > 3 || pixels < 655360 || pixels > 8294400) {
        usage('Invalid GPT Image 2/2.5 size: edges must be multiples of 16, at most 3840; aspect ratio at most 3:1; total pixels 655360..8294400.');
      }
      if (pixels > 3686400) warnings.push('GPT Image 2/2.5 output above 3686400 pixels is experimental.');
    } else if (!['1024x1024', '1536x1024', '1024x1536'].includes(request.size)) {
      usage('Older GPT Image models support 1024x1024, 1536x1024, 1024x1536, or auto.');
    }
  }
  if (values['output-compression'] !== undefined) {
    if (format === 'png') usage('--output-compression is only supported for JPEG and WebP.');
    request.output_compression = integer(values['output-compression'], 'output-compression', 0, 100);
  }
  if (values.moderation !== undefined) request.moderation = choice(values.moderation, 'moderation', ['auto', 'low']);
  if (values.user !== undefined) request.user = values.user;
  if (values.stream || values['partial-images'] !== undefined) request.stream = true;
  if (values['partial-images'] !== undefined) request.partial_images = integer(values['partial-images'], 'partial-images', 0, 3);
  const timeout = integer(values.timeout ?? '900', 'timeout', 1, 86400) * 1000;
  const images = (values.image ?? []).map(file => path.resolve(file));
  const mask = values.mask ? path.resolve(values.mask) : undefined;
  if (kind === 'edit') {
    if (!images.length || images.length > 16) usage('edit requires 1 to 16 --image files.');
    const metadata = [];
    for (const file of [...images, ...(mask ? [mask] : [])]) {
      const info = fs.statSync(file);
      if (!info.isFile() || info.size >= 50 * 1024 * 1024) usage(`Input must be a file smaller than 50 MB: ${file}`);
      const meta = await sharp(file).metadata();
      if (!['png', 'jpeg', 'webp'].includes(meta.format)) usage(`Input must be PNG, JPEG, or WebP: ${file}`);
      metadata.push(meta);
    }
    if (mask) {
      const meta = metadata[metadata.length - 1];
      if (meta.format !== 'png' || !meta.hasAlpha || meta.width !== metadata[0].width || meta.height !== metadata[0].height) usage('Mask must be a PNG with an alpha channel and the same dimensions as the first image.');
    }
    if (values['input-fidelity'] !== undefined) {
      if (request.model.startsWith('gpt-image-2')) usage('GPT Image 2/2.5 always uses high input fidelity. Omit --input-fidelity.');
      request.input_fidelity = choice(values['input-fidelity'], 'input-fidelity', ['low', 'high']);
    }
  }
  getKey();
  // Keep reference bytes with the job so later edits/deletions cannot change a queued request.
  privateDir(jobsDir);
  const id = randomUUID();
  const assetsDir = path.join(jobsDir, id);
  privateDir(assetsDir);
  const outputPaths = Array.from({ length: request.n }, (_, i) => request.n === 1 ? out : `${out.slice(0, out.length - extension.length)}-${i + 1}${extension}`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  for (const file of outputPaths) {
    if (fs.existsSync(file)) usage(`Output already exists: ${file}. Choose a new --out path.`);
    fs.accessSync(path.dirname(file), fs.constants.W_OK);
  }
  const copies = [];
  for (const [index, file] of [...images, ...(mask ? [mask] : [])].entries()) {
    const dest = path.join(assetsDir, `input-${index}${path.extname(file)}`);
    fs.copyFileSync(file, dest, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(dest, 0o600);
    copies.push(dest);
  }
  const job: Job = {
    id, kind, status: 'queued', created_at: now(), updated_at: now(),
    request, timeout_ms: timeout, output_paths: outputPaths,
    images: copies.slice(0, images.length), ...(mask ? { mask: copies.at(-1) } : {}),
    outputs: [], previews: [], warnings,
  };
  atomicJSON(jobFile(id), job);
  try {
    const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '__worker', id], { detached: true, stdio: 'ignore' });
    await once(child, 'spawn');
    child.unref();
  } catch (error) {
    job.status = 'failed';
    job.error = { message: `Unable to start worker: ${errorMessage(error)}` };
    job.finished_at = job.updated_at = now();
    atomicJSON(jobFile(id), job);
    throw error;
  }
  if (!values.wait) {
    if (values.json) console.log(JSON.stringify({ id, status: 'queued', output_paths: outputPaths }));
    else console.log(`Started ${id}\nCheck: imgen status ${id} --watch`);
    return;
  }
  console.error(`Waiting for ${id}. Interrupting this command will not cancel it.`);
  while (true) {
    const current = readJob(id);
    if (!activeStates.has(current.status)) {
      printJob(current, values.json);
      if (current.status !== 'completed') process.exitCode = 1;
      return;
    }
    await sleep(500);
  }
}

async function worker(id: string) {
  const job = readJob(id);
  if (job.status !== 'queued') throw new Error('Worker can only start a queued job.');
  const save = () => { job.updated_at = now(); atomicJSON(jobFile(id), job); };
  job.pid = process.pid;
  job.status = 'running';
  job.started_at = now();
  save();
  const heartbeat = setInterval(save, 5000);
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(new Error('API request deadline exceeded')), job.timeout_ms);
  const stop = () => controller.abort(new Error('Worker was interrupted'));
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);
  let apiKey: string | undefined;
  try {
    apiKey = getKey();
    const client = new OpenAI({ apiKey, baseURL: 'https://api.openai.com/v1', maxRetries: 0, timeout: job.timeout_ms });
    if (job.kind === 'respond') {
      const request = job.request as ResponseRequest;
      const content: Array<Record<string, unknown>> = [{
        type: 'input_text',
        text: `${request.prompt}\n\nUse the image generation tool and return one finished image. Saved @character references define recurring identities; preserve each person's recognizable facial features and appearance while following the requested scene.`,
      }];
      for (const item of request.characters) {
        content.push({ type: 'input_text', text: `The following ${item.images.length} reference image${item.images.length === 1 ? '' : 's'} define @${item.name}.${item.description ? ` ${item.description}` : ''}` });
        for (const file of item.images) {
          const meta = await sharp(file).metadata();
          const mime = meta.format === 'jpeg' ? 'image/jpeg' : `image/${meta.format}`;
          content.push({ type: 'input_image', detail: 'high', image_url: `data:${mime};base64,${fs.readFileSync(file).toString('base64')}` });
        }
      }
      const tool: Record<string, unknown> = {
        type: 'image_generation', model: request.image_model, action: request.action,
        size: request.size, quality: request.quality, background: request.background,
        output_format: request.output_format,
        ...(request.output_compression !== undefined ? { output_compression: request.output_compression } : {}),
        ...(request.moderation ? { moderation: request.moderation } : {}),
      };
      const response = await client.responses.create({
        model: request.model,
        input: [{ role: 'user', content }] as never,
        tools: [tool] as never,
        ...(request.previous_response_id ? { previous_response_id: request.previous_response_id } : {}),
      }, { signal: controller.signal });
      job.response_id = response.id;
      job.request_id = response._request_id;
      job.usage = response.usage;
      const calls = response.output.filter(item => item.type === 'image_generation_call');
      for (const [index, call] of calls.entries()) {
        const file = job.output_paths[index];
        if (!file) throw new Error('Responses API returned more images than requested.');
        job.outputs.push(await writeImage(file, call.result ?? undefined));
      }
      if (job.outputs.length !== 1) throw new Error(`Expected 1 image from the Responses API, received ${job.outputs.length}.`);
      if (request.background === 'transparent' && !job.outputs[0].has_transparency) job.warnings.push(`No transparent pixels in ${job.outputs[0].path}. The prompt may have overridden the background setting.`);
      job.status = 'completed';
      return;
    }
    const imageRequest = job.request as ImageRequest;
    let images: File[] = [];
    let mask: File | undefined;
    if (job.kind === 'edit') {
      images = await Promise.all(job.images.map(async file => {
        const meta = await sharp(file).metadata();
        return toFile(fs.createReadStream(file), path.basename(file), { type: `image/${meta.format}` });
      }));
      if (job.mask) mask = await toFile(fs.createReadStream(job.mask), 'mask.png', { type: 'image/png' });
    }
    const call = job.kind === 'edit'
      ? client.images.edit({ ...imageRequest, image: images, mask }, { signal: controller.signal })
      : client.images.generate(imageRequest, { signal: controller.signal });
    const { data, request_id: requestId } = await call.withResponse();
    job.request_id = requestId;
    save();
    if (Symbol.asyncIterator in data) {
      for await (const event of data) {
        job.last_event = event.type;
        if (event.type === 'image_generation.partial_image' || event.type === 'image_edit.partial_image') {
          const file = path.join(jobsDir, id, `preview-${job.previews.length + 1}.${event.output_format || imageRequest.output_format}`);
          job.previews.push(await writeImage(file, event.b64_json));
        } else if (event.type === 'image_generation.completed' || event.type === 'image_edit.completed') {
          const file = job.output_paths[job.outputs.length];
          if (!file) throw new Error('API returned more images than requested.');
          job.outputs.push(await writeImage(file, event.b64_json));
          if (event.usage) job.usage = event.usage;
          job.actual = { size: event.size, quality: event.quality, background: event.background, output_format: event.output_format };
        }
        save();
      }
    } else {
      job.usage = data.usage;
      job.actual = { size: data.size, quality: data.quality, background: data.background, output_format: data.output_format };
      for (const [index, image] of (data.data ?? []).entries()) {
        const file = job.output_paths[index];
        if (!file) throw new Error('API returned more images than requested.');
        job.outputs.push(await writeImage(file, image.b64_json));
        save();
      }
    }
    if (job.outputs.length !== imageRequest.n) throw new Error(`Expected ${imageRequest.n} images, received ${job.outputs.length}. Saved outputs remain available.`);
    if (imageRequest.background === 'transparent') {
      for (const image of job.outputs) if (!image.has_transparency) job.warnings.push(`No transparent pixels in ${image.path}. The prompt may have overridden the background setting.`);
    }
    job.status = 'completed';
  } catch (error) {
    job.status = 'failed';
    const message = controller.signal.aborted ? `${errorMessage(controller.signal.reason)}. OpenAI may still have processed the request; it was not retried.` : errorMessage(error);
    job.error = { message: apiKey ? message.split(apiKey).join('[REDACTED]') : message };
    if (error instanceof OpenAI.APIError) {
      job.error.status = error.status;
      job.error.code = error.code;
      if (error.requestID) job.request_id = error.requestID;
    }
  } finally {
    clearInterval(heartbeat);
    clearTimeout(deadline);
    process.off('SIGTERM', stop);
    process.off('SIGINT', stop);
    job.finished_at = now();
    save();
  }
}

async function writeImage(file: string, base64: string | undefined): Promise<SavedImage> {
  if (typeof base64 !== 'string' || !base64.length) throw new Error('API returned no image data.');
  const bytes = Buffer.from(base64, 'base64');
  const image = sharp(bytes);
  const meta = await image.metadata();
  const alpha = await image.ensureAlpha().extractChannel('alpha').raw().toBuffer();
  let transparent = 0;
  let nonOpaque = 0;
  for (const value of alpha) { if (value === 0) transparent++; if (value < 255) nonOpaque++; }
  // An exclusive hard link publishes a complete file without clobbering another job's output.
  const temp = path.join(path.dirname(file), `.imgen-${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temp, bytes, { flag: 'wx', mode: 0o600 });
    fs.linkSync(temp, file);
  } finally {
    fs.rmSync(temp, { force: true });
  }
  return {
    path: file, width: meta.width, height: meta.height, format: meta.format, bytes: bytes.length,
    has_alpha: Boolean(meta.hasAlpha), has_transparency: nonOpaque > 0,
    fully_transparent_fraction: transparent / alpha.length,
  };
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (!command || command === 'help' || command === '--help' || command === '-h') return console.log(help);
  if (command === '__worker') return worker(args[0]);
  if (command !== 'generate' && command !== 'edit' && command !== 'respond' && command !== 'character' && command !== 'status' && command !== 'login') usage(`Unknown command: ${command}. Run imgen --help.`);
  let parsed;
  try { parsed = parseArgs({ args, options: options[command], allowPositionals: true, strict: true }); }
  catch (error) { usage(errorMessage(error)); }
  const { positionals } = parsed;
  const values = parsed.values as Values;
  if (values.help) return console.log(help);
  if (command === 'login') {
    if (positionals.length) usage('Do not pass your API key as an argument. Run imgen login.');
    return login(values);
  }
  if (command === 'status') return status(values, positionals);
  if (command === 'character') return character(values, positionals);
  if (command === 'respond') return submitResponse(values, positionals);
  return submit(command, values, positionals);
}

main().catch((error: unknown) => {
  const message = errorMessage(error);
  if (process.argv.includes('--json')) console.error(JSON.stringify({ error: { message } }));
  else console.error(`imgen: ${message}`);
  process.exitCode = error instanceof Error && 'exitCode' in error && typeof error.exitCode === 'number' ? error.exitCode : 1;
});
