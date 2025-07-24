import { randomUUID } from 'crypto';
import { Config } from '../config/config.js';
import { Logger } from '../core/logger.js';

export async function autoSaveChatIfEnabled(config: Config) {
  console.log('Attempting to autosave...');
  if (!config.getAutosaveEnabled()) {
    console.log('Autosave is not enabled, skipping.');
    return;
  }

  console.log('Autosave is enabled, proceeding...');
  const logger = new Logger(config.getSessionId());
  await logger.initialize();

  const chat = config.getGeminiClient()?.getChat();
  if (!chat) {
    console.log('Autosave: No chat client found.');
    return;
  }

  const history = chat.getHistory();
  if (history.length === 0) {
    console.log('Autosave: No history found to save.');
    return;
  }

  let tag = config.getResumedChatTag();
  if (!tag) {
    tag = randomUUID();
    config.setResumedChatTag(tag);
    console.log(`Autosave: No resumed tag found, created new tag: ${tag}`);
  } else {
    console.log(`Autosave: Using resumed tag: ${tag}`);
  }

  await logger.saveCheckpoint(history, tag);
  console.log(`Autosave: Successfully saved checkpoint with tag: ${tag}`);
}
