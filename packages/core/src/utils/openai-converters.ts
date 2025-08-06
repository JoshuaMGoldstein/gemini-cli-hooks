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
import { isThinkingSupported } from '../core/modelCheck.js';

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

export function toGeminiRequest(request: GenerateContentParameters): {
  messages: ChatCompletionMessageParam[];
  model: string;
  temperature: number;
  top_p: number;
  tools?: ChatCompletionTool[];
  tool_choice?: 'auto';
  reasoning?: Record<string, unknown>;
} {
  const { contents, config } = request;
  const tools = config?.tools;
  const messages: ChatCompletionMessageParam[] = [];
  const history = (
    Array.isArray(contents) ? contents : [contents]
  ).filter((content): content is Content => typeof content === 'object');

  for (var h=0; h<history.length; h++) {
    let content =history[h];
    let priorcontent = h>0?history[h-1]:null;
    
    const functionCallParts = (priorcontent && priorcontent.role == 'model')?priorcontent.parts?.filter((part: any) => 'functionCall' in part) || [] : [];
    const functionResponseParts =
      content.parts?.filter((part: any) => 'functionResponse' in part) || [];

    if (functionResponseParts.length > 0) {
      for (const responsePart of functionResponseParts as any[]) {
        const functionResponse = responsePart.functionResponse;
        const matchingCall = functionCallParts.find(
          (callPart: any) =>
            callPart.functionCall.id === functionResponse.id,
        );

        const functionCall = (matchingCall as any)?.functionCall;

        messages.push({
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: functionResponse.id,
              type: 'function',
              function: {
                name: functionCall?.name || functionResponse.name,
                arguments: functionCall?.args
                  ? JSON.stringify(functionCall.args)
                  : '{}',
              },
            },
          ],
        });

        messages.push({
          role: 'tool',
          tool_call_id: functionResponse.id,
          content: JSON.stringify(functionResponse.response),
        });
      }
    } else if (functionCallParts.length > 0) {
      // This block is intentionally left empty.
      // We do not want to generate an assistant message for a tool call
      // unless we have the corresponding tool response.
    } else {
      // Regular user or model message
      messages.push({
        role: content.role === 'model' ? 'assistant' : 'user',
        content: toOpenAiContent(content.parts as Part[]),
      });
    }
  }

  const openAiTools = tools ? toOpenAiTools(tools as Tool[]) : undefined;
  const reasoning = isThinkingSupported(request.model || '')
    ? {}
    : undefined;

  return {
    messages,
    model: request.model || '',
    temperature: config?.temperature || 0,
    top_p: config?.topP || 1,
    tools: openAiTools,
    tool_choice: openAiTools ? 'auto' : undefined,
    reasoning,
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
