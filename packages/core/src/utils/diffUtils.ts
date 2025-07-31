/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import chalk from 'chalk';

export function formatFileDiff(diffContent: string): string {
  const lines = diffContent.split('\n');
  const formattedLines = lines.map((line) => {
    if (line.startsWith('+')) {
      return chalk.green(line);
    } else if (line.startsWith('-')) {
      return chalk.red(line);
    } else if (line.startsWith('@@')) {
      return chalk.cyan(line);
    }
    return line;
  });
  return formattedLines.join('\n');
}
