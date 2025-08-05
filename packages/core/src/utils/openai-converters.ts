/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  GenerateContentParameters,
  Part,
  Content,
} from '@google/genai';
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
} from 'openai/resources';

function toOpenAiContent(parts: Part[]): string {
  return parts.map((part) => part.text).join('');
}

export function toGeminiRequest(request: GenerateContentParameters): {
  messages: ChatCompletionMessageParam[];
  model: string;
  temperature: number;
  top_p: number;
} {
  const { contents, config } = request;
  const messages: ChatCompletionMessageParam[] = (
    Array.isArray(contents) ? contents : [contents]
  )
    .filter((content): content is Content => typeof content === 'object')
    .map((content: Content) => ({
      role: content.role === 'model' ? 'assistant' : 'user',
      content: toOpenAiContent(content.parts as Part[]),
    }));

  return {
    messages,
    model: request.model || '',
    temperature: config?.temperature || 0,
    top_p: config?.topP || 1,
  };
}

export function toGeminiResponse(
  response: ChatCompletion | ChatCompletionChunk,
): GenerateContentResponse {
  const choice = response.choices[0];
  const part: Part = {};

  if ('message' in choice) {
    part.text = choice.message.content || '';
  } else if (choice.delta) {
    part.text = choice.delta.content || '';
  }

  return {
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: [part],
        },
      },
    ],
    promptFeedback: {
      safetyRatings: [],
    },
    text: part.text || '',
    data: '',
    functionCalls: [],
    executableCode: '',
    codeExecutionResult: '',
  };
}