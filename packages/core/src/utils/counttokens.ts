import { getEncoding } from 'js-tiktoken';
import { Content, Part } from '@google/genai';

const tokenizer = getEncoding('cl100k_base');
const countTokens = (text: string) => tokenizer.encode(text).length;

// Helper function to count tokens in different part types
export function countTokensInPart(part: Part): number {
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

export function countContentTokens (content:Content):number {
  let tokens = countTokens(content.role || ''); // Count role tokens
  tokens += (content.parts || []).reduce(
    (partAcc, part) => partAcc + countTokensInPart(part),
    0
  );
  return tokens;
}

// Count tokens for role + all parts
export function countHistoryTokens(history:Content[]):number {
    return history.reduce(
        (acc:number, content:Content) => {
            return acc + countContentTokens(content);
        },
        0
    );
}