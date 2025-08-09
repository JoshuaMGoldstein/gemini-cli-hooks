import { randomUUID } from 'crypto';
import { Config } from '../config/config.js';
import { Logger } from '../core/logger.js';
import { Content, Part } from '@google/genai';
import { countContentTokens,countHistoryTokens,countTokensInPart } from './counttokens.js';

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


  if (autosaveSettings) {
    const {      
      truncateafter,
      truncatenewtag,      
      minstartingtokens,
    } = autosaveSettings;

   
    let historyTokens = countHistoryTokens(history);
    

    let tag = config.getResumedChatTag();

    if (historyTokens > truncateafter) {
      
      if (truncatenewtag || !tag) {
        tag = randomUUID();
        config.setResumedChatTag(tag);
      }

      const targetTokens =
        (autosaveSettings.truncateby && autosaveSettings.truncateby > 1)
          ? autosaveSettings.truncateby
          : truncateafter * (autosaveSettings.truncateby || 0.8);

      const tokenCounts = history.map(content => {
        let tokens = countContentTokens(content)
        return { content, tokens };
      });

      let keptTokens = 0;
      let startingHistory: Content[] = [];
      for (const { content, tokens } of tokenCounts) {
        // Ensure we don't add a huge message that blows past the budget,
        // unless it's the very first or second message
        //Should we allowe third and fourth messages for compression summaries? Not sure.
        if (keptTokens + tokens > minstartingtokens && startingHistory.length > 1) {
          break;
        }
        startingHistory.push(content);
        keptTokens += tokens;
        // Stop once we have met the minimum requirement.
        if (keptTokens >= minstartingtokens) {
          break;
        }
      }

      let additionalCompressed = autosaveSettings.additionalcompressed || 20000;
      let compressionCharLimit = autosaveSettings.compressioncharlimit || 560;      
      let endingHistory: Content[] = [];
      const tokensToKeepAtEnd = targetTokens - keptTokens;
      const tokensToKeepAtEndPlusCompressed = targetTokens - keptTokens + additionalCompressed
      let tokensCountedAtEnd = 0;
      // Start from the end, but stop before overlapping with startingHistory
      for (let i = tokenCounts.length - 1; i >= startingHistory.length; i--) {
        const { content, tokens } = tokenCounts[i];

        if (tokensCountedAtEnd + tokens <= tokensToKeepAtEnd) {
          endingHistory.unshift(content);
          tokensCountedAtEnd += tokens;
        } else {
          //Keep going and add additional compressed content.                    
          if(content.parts) {
            for(var j=0; j<content.parts.length;j++) {
              let part = content.parts[j];
              
              let truncationPrefixString = `[truncated to ${compressionCharLimit} chars. Run command again for full content]`;
              let truncObj:any = null; //Object to truncate the keys of
              if(part.functionResponse && part.functionResponse?.response ) {
                truncObj = part.functionResponse.response;    //Truncate the function response          
              } else if(part.functionCall && part.functionCall?.args) {
                truncObj = part.functionCall.args; //Truncate arugments to write file etc
              } else if(part.text && part.text.length > (compressionCharLimit+truncationPrefixString.length) ) {
                part.text = truncationPrefixString+part.text.slice(0,compressionCharLimit);
              }
              if(truncObj) {
                let keys = Object.keys(truncObj);
                for(var k=0; k<keys.length; k++) {
                  let key = keys[k];
                  if(typeof truncObj[key] === 'string' && truncObj[key].length > (compressionCharLimit+truncationPrefixString.length) ) {
                    truncObj[key] = truncationPrefixString+truncObj[key].slice(0,compressionCharLimit);
                  }
                }
              }
            }            
            
            let newTokens = countContentTokens(content); //the new token count of the compressed content.
            if(tokensCountedAtEnd + newTokens <= tokensToKeepAtEndPlusCompressed) {
              //IT fits, so add the compressed content
              endingHistory.unshift(content); 
              tokensCountedAtEnd += newTokens;
            } else {
              break; //We couldnt add this, even compressed, so we're done
            }
          }
        }
      }
      
      history = [...startingHistory, ...endingHistory];
      chat.setHistory(history);
      
    } 
  }

  let tag = config.getResumedChatTag();
  if (!tag) {
    tag = randomUUID();
    config.setResumedChatTag(tag);
  }
  
  await logger.saveCheckpoint(history, tag);
}