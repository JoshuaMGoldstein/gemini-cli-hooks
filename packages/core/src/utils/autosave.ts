import { randomUUID } from 'crypto';
import { Config } from '../config/config.js';
import { Logger } from '../core/logger.js';

export async function autoSaveChatIfEnabled(config: Config) {
  if (!config.getAutosaveEnabled()) {
    return;
  }

  const logger = new Logger(config.getSessionId());
  await logger.initialize();

  const chat = config.getGeminiClient()?.getChat();
  if (!chat) {
    return;
  }

  const history = chat.getHistory();
  if (history.length === 0) {
    return;
  }

  let tag = config.getResumedChatTag();
  if (!tag) {
    tag = randomUUID();
    config.setResumedChatTag(tag);
  }

  await logger.saveCheckpoint(history, tag);
}
