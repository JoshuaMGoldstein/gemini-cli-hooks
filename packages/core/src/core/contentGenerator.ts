/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  GoogleGenAI,
  Part,
  Content
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { Config } from '../config/config.js';
import { getEffectiveModel } from './modelCheck.js';
import { UserTierId } from '../code_assist/types.js';
import OpenAI from 'openai';
import { Tiktoken, getEncoding } from 'js-tiktoken';
import {
  toGeminiRequest,
  toGeminiResponse,
} from '../utils/openai-converters.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  userTier?: UserTierId;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  CLOUD_SHELL = 'cloud-shell',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
  proxy?: string | undefined;
};

class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private tokenizer: Tiktoken;
  private config: Config;

  constructor(apiKey: string, baseUrl: string, config: Config) {
    this.openai = new OpenAI({
      apiKey,
      baseURL: baseUrl,
    });
    this.tokenizer = getEncoding('cl100k_base');
    this.config = config;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const mappedRequest = toGeminiRequest(request, this.config);
    if (this.config.getDebugMode()) {
      console.log('OpenAI Request URL:', this.openai.baseURL + 'chat/completions');
      console.log('OpenAI Request:', JSON.stringify(mappedRequest, null, 2));
    }
    try {
      const response = await this.openai.chat.completions.create({
        ...mappedRequest,
        stream: false,
      });
      return toGeminiResponse(response);
    } catch (error) {
      console.error(
        `Error during OpenAI API call. Request URL: ${this.openai.baseURL}/chat/completions`,
      );
      throw error;
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const mappedRequest = toGeminiRequest(request, this.config);
    if (this.config.getDebugMode()) {
      console.log('OpenAI Request URL:', this.openai.baseURL + 'chat/completions');
      console.log('OpenAI Request:', JSON.stringify(mappedRequest, null, 2));
    }
    const stream = await this.openai.chat.completions.create({
      ...mappedRequest,
      stream: true,
    });

    async function* mapStream(): AsyncGenerator<GenerateContentResponse> {
      for await (const chunk of stream) {
        yield toGeminiResponse(chunk);
      }
    }

    return mapStream();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = Array.isArray(request.contents)?request.contents:[request.contents];//Array.isArray(request) ? request : [request];
    let totalTokens = 0;
    let textArray = [];
    for (const content of contents) {
      if(typeof content === 'string') {
        textArray.push(content);
      } else if ((content as Part).text) {
        textArray.push((content as Part).text || '')
      } else {
        let parts = (content as Content).parts;
        if(parts) {
          for(var p=0; p<parts.length; p++) {
            if(parts[p].text) {
              textArray.push(parts[p].text || '')
            }
          }
        }        
      }
    }
    totalTokens += this.tokenizer.encode(textArray.join(' ')).length;
    return { totalTokens };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent is not supported for OpenAI models.');
  }
}

export function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
): ContentGeneratorConfig {
  const geminiApiKey = process.env.GEMINI_API_KEY || undefined;
  const googleApiKey = process.env.GOOGLE_API_KEY || undefined;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT || undefined;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION || undefined;

  // Use runtime model from config if available; otherwise, fall back to parameter or default
  const effectiveModel = config.getModel() || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.CLOUD_SHELL
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;
    getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
      contentGeneratorConfig.proxy,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const openAiApiKey = gcConfig.getOpenAiApiKey();
  const openAiBaseUrl = gcConfig.getOpenAiBaseUrl();

  if (openAiApiKey && openAiBaseUrl) {
    return new OpenAIContentGenerator(openAiApiKey, openAiBaseUrl, gcConfig);
  }

  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };
  if (
    config.authType === AuthType.LOGIN_WITH_GOOGLE ||
    config.authType === AuthType.CLOUD_SHELL
  ) {
    return createCodeAssistContentGenerator(
      httpOptions,
      config.authType,
      gcConfig,
      sessionId,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}