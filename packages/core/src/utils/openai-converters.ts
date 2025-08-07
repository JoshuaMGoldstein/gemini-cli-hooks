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
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources';
import { isThinkingSupported } from '../core/modelCheck.js';
import { Config } from '../config/config.js';

function toOpenAiContent(parts: Part[]): string {
  if (!Array.isArray(parts)) {
    return '';
  }
  return parts.map((part) => part?.text || '').join('');
}

export function toOpenAiTools(tools: Tool[]): ChatCompletionTool[] {
  // Helper function to recursively convert schema type values to lowercase
  const convertSchemaTypes = (schema: any): any => {
    if (!schema) {
      return schema;
    }

    const newSchema = { ...schema };

    if (newSchema.type) {
      newSchema.type = newSchema.type.toLowerCase();
    }

    if (newSchema.minLength && typeof newSchema.minLength === 'string') {
      newSchema.minLength = parseInt(newSchema.minLength, 10);
    }

    if (newSchema.minItems && typeof newSchema.minItems === 'string') {
      newSchema.minItems = parseInt(newSchema.minItems, 10);
    }

    if (newSchema.properties) {
      newSchema.properties = Object.entries(newSchema.properties).reduce(
        (acc, [key, value]) => {
          acc[key] = convertSchemaTypes(value);
          return acc;
        },
        {} as { [key: string]: any },
      );
    }

    if (newSchema.items) {
      newSchema.items = convertSchemaTypes(newSchema.items);
    }

    return newSchema;
  };

  return tools.flatMap(tool =>
    (tool.functionDeclarations || []).map(func => ({
      type: 'function' as const,
      function: {
        name: func.name || '',
        description: func.description || '',
        parameters: convertSchemaTypes(func.parameters),
      },
    }))
  );
}

export function toGeminiRequest(
  request: GenerateContentParameters,
  config: Config,
): {
  messages: ChatCompletionMessageParam[];
  model: string;
  temperature: number;
  top_p: number;
  tools?: ChatCompletionTool[];
  tool_choice?: 'auto';
  reasoning?: Record<string, unknown>;
  response_format?: { type: 'json_object' };
} {
  const { contents } = request;
  const tools = request.config?.tools;
  const messages: ChatCompletionMessageParam[] = [];
  const history = (
    Array.isArray(contents) ? contents : [contents]
  ).filter((content): content is Content => typeof content === 'object');

  // Build OpenAI messages with strict tool-call pairing and merge text + tool_calls
  for (let h = 0; h < history.length; h++) {
    const content = history[h];

    const functionResponseParts =
      content.parts?.filter((part: any) => 'functionResponse' in part) || [];
    const functionCallPartsInThisMessage =
      content.parts?.filter((part: any) => 'functionCall' in part) || [];

    // If current message has functionResponse(s), we must find matching functionCall
    if (functionResponseParts.length > 0) {
      for (const responsePart of functionResponseParts as any[]) {
        const functionResponse = responsePart.functionResponse;

        // Search backwards to find the matching functionCall by id
        let matchingCall: any | undefined;
        for (let k = h - 1; k >= 0 && !matchingCall; k--) {
          const prev = history[k];
          if (!prev?.parts) continue;
          matchingCall = prev.parts.find(
            (p: any) => p.functionCall && p.functionCall.id === functionResponse.id,
          );
        }

        const functionCall = matchingCall?.functionCall as
          | { name?: string; args?: unknown }
          | undefined;

        // Create assistant tool_call followed immediately by tool
        messages.push({
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: functionResponse.id,
              type: 'function',
              function: {
                name: functionCall?.name || functionResponse.name || '',
                arguments:
                  functionCall?.args !== undefined
                    ? JSON.stringify(functionCall.args)
                    : '{}',
              },
            },
          ],
        });

        messages.push({
          role: 'tool',
          tool_call_id: functionResponse.id,
          content: JSON.stringify(functionResponse.response ?? {}),
        });
      }
      continue;
    }

    // If message has functionCall(s) but no response yet, do NOT emit anything now
    if (functionCallPartsInThisMessage.length > 0) {
      continue;
    }

    // Otherwise emit regular turn (merge any text parts)
    const text = toOpenAiContent(content.parts as Part[]);
    messages.push({
      role: content.role === 'model' ? 'assistant' : 'user',
      content: text,
    });
  }

  const openAiTools = tools ? toOpenAiTools(tools as Tool[]) : undefined;
  let reasoning: Record<string, unknown> | undefined;
  if (isThinkingSupported(request.model || '')) {
    reasoning = {};
    if (config.getReasoningMax()) {
      reasoning.max_tokens = config.getReasoningMax();
    }
  }
  const response_format =
    request.config?.responseMimeType === 'application/json'
      ? { type: 'json_object' as const }
      : undefined;

  // Second pass: merge consecutive assistant messages, but NEVER across a tool boundary.
  // Specifically, do not merge if the previous assistant had tool_calls and the next
  // message is role: 'tool' (they must remain adjacent and unbroken).
  const mergedMessages: ChatCompletionMessageParam[] = [];
  for (let i = 0; i < messages.length; i++) {
    const curr = messages[i];

    if (curr.role !== 'assistant') {
      mergedMessages.push(curr);
      continue;
    }

    // If this assistant has tool_calls, and next message is tool, keep as-is
    const next = messages[i + 1];
    if (curr.role === 'assistant' && (curr as any).tool_calls && next?.role === 'tool') {
      mergedMessages.push(curr);
      continue;
    }

    // Otherwise, merge with following assistant messages that also have no immediate tool after them
    let mergedContent: string[] = [];
    let mergedToolCalls: any[] = [];

    const take = (msg: ChatCompletionAssistantMessageParam) => {
      if (typeof msg.content === 'string' && msg.content) mergedContent.push(msg.content);
      if ((msg as any).tool_calls) mergedToolCalls.push(...(msg as any).tool_calls);
    };

    take(curr as ChatCompletionAssistantMessageParam);

    let j = i + 1;
    while (j < messages.length && messages[j].role === 'assistant') {
      const prevOfJ = messages[j - 1];
      const nextOfJ = messages[j + 1];
      const isToolBoundary = (prevOfJ as any)?.tool_calls && nextOfJ?.role === 'tool';
      if (isToolBoundary) break; // don't cross assistant-tool pair
      take(messages[j] as ChatCompletionAssistantMessageParam);
      j++;
    }

    mergedMessages.push({
      role: 'assistant',
      content: mergedContent.length ? mergedContent.join('\n') : null,
      tool_calls: mergedToolCalls.length ? mergedToolCalls : undefined,
    } as ChatCompletionAssistantMessageParam);

    i = j - 1; // advance outer loop
  }

  return {
    messages: mergedMessages,
    model: request.model || '',
    temperature: request.config?.temperature || 0,
    top_p: request.config?.topP || 1,
    tools: openAiTools,
    tool_choice: openAiTools ? 'auto' : undefined,
    reasoning,
    response_format,
  };
}

const partialToolCalls: { [key: string]: { id: string; name: string; arguments: string } } = {};

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
        const toolCallIndex = toolCall.index;
        if (toolCall.id) {
          if (!partialToolCalls[toolCallIndex]) {
            partialToolCalls[toolCallIndex] = { id: '', name: '', arguments: '' };
          }
          partialToolCalls[toolCallIndex].id = toolCall.id;
        }
        if (toolCall.function?.name) {
          partialToolCalls[toolCallIndex].name = toolCall.function.name;
        }
        if (toolCall.function?.arguments) {
          partialToolCalls[toolCallIndex].arguments += toolCall.function.arguments;
        }
      }
    }
  }

  if (choice.finish_reason === 'tool_calls') {
    for (const key in partialToolCalls) {
      const toolCall = partialToolCalls[key];
      functionCalls.push({
        name: toolCall.name,
        args: JSON.parse(toolCall.arguments),
        id: toolCall.id,
      });
    }
    // Clear the partial tool calls for the next response
    for (const key in partialToolCalls) {
      delete partialToolCalls[key];
    }
  }

  const parts: Part[] = [];
  if (part.text) {
    parts.push({ text: part.text });
  }

  if (functionCalls.length > 0) {
    for (const functionCall of functionCalls) {
      parts.push({ functionCall });
    }
  }

  return {
    candidates: [
      {
        index: 0,
        content: {
          role: 'model',
          parts: parts,
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
