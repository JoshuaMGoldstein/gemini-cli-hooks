import { randomUUID } from 'crypto';
import { Config } from '../config/config.js';
import { Logger } from '../core/logger.js';
import { Content, Part } from '@google/genai';
import { getEncoding } from 'js-tiktoken';

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

  let history = chat.getHistory();
  if (history.length === 0) {
    return;
  }

  const autosaveSettings = config.getAutosaveSettings();
  console.log(`[AUTOSAVE DEBUG] Settings: ${JSON.stringify(autosaveSettings)}`);

  if (autosaveSettings) {
    const {
      compressafter,
      truncateafter,
      truncatenewtag,
      compressnewtag,
      minstartingtokens,
    } = autosaveSettings;

    const tokenizer = getEncoding('cl100k_base');
    const countTokens = (text: string) => tokenizer.encode(text).length;
    
    // Helper function to count tokens in different part types
    const countTokensInPart = (part: Part): number => {
      if ('text' in part) {
        return countTokens(part.text || '');
      } else if ('inlineData' in part) {
        // Count as image (1066 tokens for Gemini 1.5 Pro)
        return 1066;
      } else if ('functionCall' in part) {
        return countTokens(JSON.stringify(part.functionCall));
      } else if ('functionResponse' in part) {
        return countTokens(JSON.stringify(part.functionResponse));
      }
      return 0;
    };

    // Count tokens for role + all parts
    const historyTokens = history.reduce(
      (acc, content) => {
        let tokens = countTokens(content.role || ''); // Count role tokens
        tokens += (content.parts || []).reduce(
          (partAcc, part) => partAcc + countTokensInPart(part),
          0
        );
        return acc + tokens;
      },
      0
    );

    console.log(`[AUTOSAVE DEBUG] Total Tokens: ${historyTokens}`);
    console.log(`[AUTOSAVE DEBUG] Truncate After: ${truncateafter}`);
    console.log(`[AUTOSAVE DEBUG] Compress After: ${compressafter}`);

    let tag = config.getResumedChatTag();

    if (historyTokens > truncateafter) {
      console.log('[AUTOSAVE DEBUG] Truncating history...');
      if (truncatenewtag || !tag) {
        tag = randomUUID();
        config.setResumedChatTag(tag);
      }

      const targetTokens =
        (autosaveSettings.truncateby && autosaveSettings.truncateby > 1)
          ? autosaveSettings.truncateby
          : truncateafter * (autosaveSettings.truncateby || 0.8);

      const tokenCounts = history.map(content => {
        let tokens = countTokens(content.role || '');
        tokens += (content.parts || []).reduce(
          (partAcc, part) => partAcc + countTokensInPart(part),
          0
        );
        return { content, tokens };
      });

      let keptTokens = 0;
      let startingHistory: Content[] = [];
      for (const { content, tokens } of tokenCounts) {
        startingHistory.push(content);
        keptTokens += tokens;
        if (keptTokens >= minstartingtokens) {
          break;
        }
      }

      let endingHistory: Content[] = [];
      let tokensToKeepAtEnd = targetTokens - minstartingtokens;
      let tokensCountedAtEnd = 0;
      // Start from the end, but stop before overlapping with startingHistory
      for (let i = tokenCounts.length - 1; i >= startingHistory.length; i--) {
        const { content, tokens } = tokenCounts[i];
        if (tokensCountedAtEnd + tokens <= tokensToKeepAtEnd) {
          endingHistory.unshift(content);
          tokensCountedAtEnd += tokens;
        } else {
          break;
        }
      }
      
      history = [...startingHistory, ...endingHistory];
      chat.setHistory(history);
      console.log(`[AUTOSAVE DEBUG] History truncated to ${history.length} entries.`);
    } else if (historyTokens > compressafter) {
      console.log('[AUTOSAVE DEBUG] Compressing history...');
      if (compressnewtag || !tag) {
        tag = randomUUID();
        config.setResumedChatTag(tag);
      }
      // Placeholder for actual compression logic
      const compressedHistory = history; // Replace with real compression
      history = compressedHistory;
      chat.setHistory(history);
    } else {
      console.log('[AUTOSAVE DEBUG] No action taken.');
    }
  }

  let tag = config.getResumedChatTag();
  if (!tag) {
    tag = randomUUID();
    config.setResumedChatTag(tag);
  }
  console.log(`[AUTOSAVE DEBUG] Saving checkpoint with tag: ${tag}`);
  await logger.saveCheckpoint(history, tag);
}