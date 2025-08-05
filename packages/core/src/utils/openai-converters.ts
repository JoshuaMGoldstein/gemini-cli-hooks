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
  Tool,
  FunctionCall,
} from '@google/genai';
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources';

function toOpenAiContent(parts: Part[]): string {
  if (!Array.isArray(parts)) {
    return '';
  }
  return parts.map((part) => part?.text || '').join('');
}

export function toOpenAiTools(tools: Tool[]): ChatCompletionTool[] {
  return tools.flatMap(tool =>
    (tool.functionDeclarations || []).map(func => ({
      type: 'function' as const,
      function: {
        name: func.name || '',
        description: func.description || '',
        parameters: func.parameters as any,
      },
    }))
  );
}

export function toGeminiRequest(request: GenerateContentParameters & { tools?: Tool[] }): {
  messages: ChatCompletionMessageParam[];
  model: string;
  temperature: number;
  top_p: number;
  tools?: ChatCompletionTool[];
  tool_choice?: 'auto';
} {
  const { contents, config, tools } = request;
  const messages: ChatCompletionMessageParam[] = (
    Array.isArray(contents) ? contents : [contents]
  )
    .filter((content): content is Content => typeof content === 'object')
    .map((content: Content) => {
      if (content.role === 'tool' && content.parts) {
        return {
          role: 'tool',
          tool_call_id: (content.parts[0] as any).functionResponse.id,
          content: JSON.stringify((content.parts[0] as any).functionResponse.response),
        };
      }
      return {
        role: content.role === 'model' ? 'assistant' : 'user',
        content: toOpenAiContent(content.parts as Part[]),
      };
    });

  const openAiTools = tools ? toOpenAiTools(tools) : undefined;

  return {
    messages,
    model: request.model || '',
    temperature: config?.temperature || 0,
    top_p: config?.topP || 1,
    tools: openAiTools,
    tool_choice: openAiTools ? 'auto' : undefined,
  };
}

export function toGeminiResponse(
  response: ChatCompletion | ChatCompletionChunk,
): GenerateContentResponse {
  const choice = response.choices[0];
  const part: Part = {};
  const functionCalls: FunctionCall[] = [];

  if ('message' in choice) {
    const message = choice.message;
    part.text = message.content || '';
    if (message.tool_calls) {
      for (const toolCall of message.tool_calls) {
        functionCalls.push({
          name: toolCall.function.name,
          args: JSON.parse(toolCall.function.arguments),
          id: toolCall.id,
        });
      }
    }
  } else if (choice.delta) {
    part.text = choice.delta.content || '';
    if (choice.delta.tool_calls) {
      for (const toolCall of choice.delta.tool_calls) {
        if (toolCall.function && toolCall.function.arguments) {
          functionCalls.push({
            name: toolCall.function.name,
            args: JSON.parse(toolCall.function.arguments),
            id: toolCall.id,
          });
        }
      }
    }
  }

  return {
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: [part],
        },
        finishReason: choice.finish_reason || undefined,
      },
    ],
    promptFeedback: {
      safetyRatings: [],
    },
    text: part.text || '',
    functionCalls,
    data: '',
    executableCode: '',
    codeExecutionResult: '',
  } as GenerateContentResponse;
}