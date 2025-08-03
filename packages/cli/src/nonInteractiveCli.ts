/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Config,
  ToolCallRequestInfo,
  executeToolCall,
  ToolRegistry,
  shutdownTelemetry,
  isTelemetrySdkInitialized,
  formatFileDiff,
  autoSaveChatIfEnabled,
} from '@google/gemini-cli-core';
import {
  Content,
  Part,
  FunctionCall,
  GenerateContentResponse,
} from '@google/genai';
import { spawn } from 'child_process';
import { parseAndFormatApiError } from './ui/utils/errorParsing.js';

function getResponseText(response: GenerateContentResponse): string | null {
  if (response.candidates && response.candidates.length > 0) {
    const candidate = response.candidates[0];
    if (
      candidate.content &&
      candidate.content.parts &&
      candidate.content.parts.length > 0
    ) {
      // We are running in headless mode so we don't need to return thoughts to STDOUT.
      const thoughtPart = candidate.content.parts[0];
      if (thoughtPart?.thought) {
        return null;
      }
      return candidate.content.parts
        .filter((part) => part.text)
        .map((part) => part.text)
        .join('');
    }
  }
  return null;
}


async function executeHook(command: string, data: object) {
  //console.log(`Executing tool_call hook: ${command}`);
  const jsonData = JSON.stringify(data);
  //console.log(`Piping JSON to stdin: ${jsonData}`);

  return new Promise<void>((resolve, reject) => {
    const child = spawn(command, { shell:true, stdio: ['pipe', 'inherit', 'inherit'] });

    child.stdin.write(jsonData);
    child.stdin.end();

    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Hook command failed with exit code ${code}`));
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
  });
}

export async function runNonInteractive(
  config: Config,
  input: string,
  prompt_id: string,
): Promise<void> {
  await config.initialize();
  // Handle EPIPE errors when the output is piped to a command that closes early.
  process.stdout.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EPIPE') {
      // Exit gracefully if the pipe is closed.
      process.exit(0);
    }
  });

  const geminiClient = config.getGeminiClient();
  const toolRegistry: ToolRegistry = await config.getToolRegistry();

  const chat = await geminiClient.getChat();
  const abortController = new AbortController();
  let currentMessages: Content[] = [{ role: 'user', parts: [{ text: input }] }];
  let turnCount = 0;
  try {
    while (true) {
      turnCount++;
      if (
        config.getMaxSessionTurns() > 0 &&
        turnCount > config.getMaxSessionTurns()
      ) {
        console.error(
          '\n Reached max session turns for this session. Increase the number of turns by specifying maxSessionTurns in settings.json.',
        );
        return;
      }
      const functionCalls: FunctionCall[] = [];

      const responseStream = await chat.sendMessageStream(
        {
          message: currentMessages[0]?.parts || [], // Ensure parts are always provided
          config: {
            abortSignal: abortController.signal,
            tools: [
              { functionDeclarations: toolRegistry.getFunctionDeclarations() },
            ],
          },
        },
        prompt_id,
      );

      for await (const resp of responseStream) {
        if (abortController.signal.aborted) {
          console.error('Operation cancelled.');
          return;
        }
        const textPart = getResponseText(resp);
        if (textPart) {
          process.stdout.write(textPart);

        }
        if(resp.data) {
          process.stdout.write('\n*Inline Data:*\n '+resp.data);
        }
        if(resp.codeExecutionResult) {
          process.stdout.write('\n*Code Execution Result:*\n'+resp.codeExecutionResult);
        }        
        if (resp.functionCalls) {
          functionCalls.push(...resp.functionCalls);
        }
      }

      if (functionCalls.length > 0) {
        const toolResponseParts: Part[] = [];

        for (const fc of functionCalls) {
          const callId = fc.id ?? `${fc.name}-${Date.now()}`;
          const requestInfo: ToolCallRequestInfo = {
            callId,
            name: fc.name as string,
            args: (fc.args ?? {}) as Record<string, unknown>,
            isClientInitiated: false,
            prompt_id,
          };

          const toolResponse = await executeToolCall(
            config,
            requestInfo,
            toolRegistry,
            abortController.signal,
          );

          // Display the tool's output to the user in non-interactive mode.
          let output: string;
          const toolOutput =
            toolResponse.responseParts &&
            typeof toolResponse.responseParts === 'object' &&
            'functionResponse' in toolResponse.responseParts
              ? toolResponse.responseParts.functionResponse?.response?.output
              : null;

          if (
            toolResponse.resultDisplay &&
            typeof toolResponse.resultDisplay === 'object' &&
            'fileDiff' in toolResponse.resultDisplay
          ) {
            output = formatFileDiff(toolResponse.resultDisplay.fileDiff);
          } else if (toolOutput && typeof toolOutput === 'string') {
            output = toolOutput;
          } else if (typeof toolResponse.resultDisplay === 'string') {
            output = toolResponse.resultDisplay;
          } else {
            output = JSON.stringify(toolResponse, null, 2);
          }
          //process.stdout.write(`\n*Tool Output: ${fc.name} ${JSON.stringify(fc.args)}*\n`);
          //process.stdout.write(output);
          //process.stdout.write(`\n\n`);

          if (config.getHooks()?.tool_call) {
            await executeHook(config.getHooks()!.tool_call!, {
              toolCall: requestInfo,
              toolResponse: toolResponse,
            });
          }

          if (toolResponse.error) {
            const isToolNotFound = toolResponse.error.message.includes(
              'not found in registry',
            );
            console.error(
              `Error executing tool ${fc.name}: ${toolResponse.resultDisplay || toolResponse.error.message}`,
            );
            if (!isToolNotFound) {
              process.exit(1);
            }
          }

          if (toolResponse.responseParts) {
            const parts = Array.isArray(toolResponse.responseParts)
              ? toolResponse.responseParts
              : [toolResponse.responseParts];
            for (const part of parts) {
              if (typeof part === 'string') {
                toolResponseParts.push({ text: part });
              } else if (part) {
                toolResponseParts.push(part);
              }
            }
          }
        }
        currentMessages = [{ role: 'user', parts: toolResponseParts }];
        await autoSaveChatIfEnabled(config);
      } else {
        process.stdout.write('\n'); // Ensure a final newline
        await autoSaveChatIfEnabled(config);
        return;
      }
    }
  } catch (error) {
    console.error(
      parseAndFormatApiError(
        error,
        config.getContentGeneratorConfig()?.authType,
      ),
    );
    process.exit(1);
  } finally {
    if (isTelemetrySdkInitialized()) {
      await shutdownTelemetry();
    }
  }
}
