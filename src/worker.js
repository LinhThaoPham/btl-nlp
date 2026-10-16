import {compareVietNews,benchmarkVietNews} from './vietnews-engine.js';
import {analyze,splitSentences} from './engine.js';
import {comparePrimaryMethods,rougeDetails,tuneAndEvaluate,numericBytes} from './research.js';
import {prepareResources,evaluationRows} from './nlp-client.js';
self.onmessage=async({data})=>{try{
  const started=performance.now();let result;
  if(data.action==='vietnews-single'||data.action==='vietnews-batch'){const progress=progress=>self.postMessage({id:data.id,progress});result=data.action==='vietnews-single'?await compareVietNews(data.record,data.options,progress):await benchmarkVietNews(data.records,data.options,progress);}
  else if(data.action==='batch')result=await tuneAndEvaluate(data.dataset,data.options,async row=>{
    const resources=await prepareResources(row.text,[],data.options,data.external);
    const evalRows=await evaluationRows([...splitSentences(row.text),row.reference],data.options.tokenizer);
    return {resources,evalRows:evalRows.slice(0,-1),referenceTokens:evalRows.at(-1)};
  },progress=>self.postMessage({id:data.id,progress}));
  else if(data.action==='rouge'){
    const rows=await evaluationRows([data.candidate,data.reference],data.tokenizer);result=rougeDetails(rows[0].join(' '),rows[1].join(' '));
  }else{
    const resources=await prepareResources(data.text,data.topics||[],data.options,data.external);
    if(data.action==='compare'){
      const embeddingOptions={...data.options,tokenizer:'underthesea',representation:'embedding'};
      const embeddingResources=data.options.tokenizer==='underthesea'&&data.options.representation==='embedding'?resources:await prepareResources(data.text,[],embeddingOptions,data.external);
      const embeddingReferenceTokens=data.reference?(await evaluationRows([data.reference],'underthesea'))[0]:null;
      result=comparePrimaryMethods(data.text,data.options,embeddingResources,data.reference,embeddingReferenceTokens,data.gold);
    }else{result=analyze(data.text,data.topics,data.options,resources);result.resourcesInfo={preprocessMs:resources.elapsedMs||0,processRssBytes:resources.processRssBytes||null,numericBytes:numericBytes(result)};}
  }
  self.postMessage({id:data.id,result,elapsed:performance.now()-started});
}catch(error){self.postMessage({id:data.id,error:error.message});}};
