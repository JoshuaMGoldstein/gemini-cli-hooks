/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn } from 'child_process';

export async function executeHook(command: string, data: object) {
  const jsonData = JSON.stringify(data);

  return new Promise<void>((resolve, reject) => {
    const child = spawn(command, { shell: true });

    child.stdin.write(jsonData);
    child.stdin.end();

    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Hook command failed with exit code ${code}`));
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
  });
}
