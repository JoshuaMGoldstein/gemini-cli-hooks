import { Config } from '../config/config.js';
import { Logger } from '../core/logger.js';
import * as fsPromises from 'fs/promises';
import * as path from 'path';
import { Content } from '@google/genai';

export async function loadCheckpoint(config: Config): Promise<Content[]> {
  const logger = new Logger(config.getSessionId());
  await logger.initialize();
  const geminiDir = config.getProjectTempDir();
  console.log('--resume: geminiTmpDir', geminiDir);
  if (!geminiDir) {
    return [];
  }
  try {
    const file_head = 'checkpoint-';
    const file_tail = '.json';
    const files = await fsPromises.readdir(geminiDir);
    const chatDetails: Array<{ name: string; mtime: Date }> = [];

    console.log('--resume: geminiTmpDir numfiles:', files.length);
    for (const file of files) {
      if (file.startsWith(file_head) && file.endsWith(file_tail)) {
        const filePath = path.join(geminiDir, file);
        const stats = await fsPromises.stat(filePath);
        chatDetails.push({
          name: file.slice(file_head.length, -file_tail.length),
          mtime: stats.mtime,
        });
      }
    }

    if (chatDetails.length > 0) {
      chatDetails.sort((a, b) => b.mtime.getTime() - a.mtime.getTime());
      const tag = chatDetails[0].name;
      const conversation = await logger.loadCheckpoint(tag);
      if (conversation.length > 0) {
        config.setResumedChatTag(tag);
        return conversation;
      }
    }

    console.log('--resume: no checkpoint found');
    return [];
  } catch (_err) {
    console.error('Error loading checkpoint:', _err);
    return []; // Ignore errors, just won't resume.
  }
}
