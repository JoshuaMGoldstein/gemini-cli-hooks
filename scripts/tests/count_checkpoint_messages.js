#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

// --- Hashing Logic (from packages/core/src/utils/paths.ts) ---
function getProjectHash(projectRoot) {
  return crypto.createHash('sha256').update(projectRoot).digest('hex');
}

// --- Directory Calculation ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// This script is in /scripts/tests, so the project root is two levels up.
const projectRoot = path.resolve(__dirname, '..', '..');
const projectHash = getProjectHash(projectRoot);
const tagdir = path.join(os.homedir(), '.gemini', 'tmp', projectHash);

console.log(`Scanning for checkpoints in: ${tagdir}`);

// --- Main Logic ---
async function countMessages() {
  try {
    const files = await fs.readdir(tagdir);

    if (files.length === 0) {
      console.log('No checkpoint files found.');
      return;
    }

    console.log('\n--- Checkpoint Message Counts ---');

    for (const file of files) {
      const filePath = path.join(tagdir, file);
      try {
        const stats = await fs.stat(filePath);
        if (!stats.isFile()) {
          continue; // Skip directories
        }

        const content = await fs.readFile(filePath, 'utf8');
        try {
          const history = JSON.parse(content);
          if (Array.isArray(history)) {
            console.log(`${file}: ${history.length} messages`);
          } else {
            console.log(`${file}: Not a valid history file (expected a JSON array).`);
          }
        } catch (parseErr) {
          console.log(`${file}: Error parsing JSON.`);
        }
      } catch (statErr) {
        console.error(`Error processing file ${file}:`, statErr);
      }
    }
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.error(`Error: Checkpoint directory not found at ${tagdir}`);
      console.error('Please run a session to generate checkpoints first.');
    } else {
      console.error(`Error reading directory ${tagdir}:`, err);
    }
    process.exit(1);
  }
}

countMessages();
