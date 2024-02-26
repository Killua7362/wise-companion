from transformers import AutoTokenizer,AutoConfig,BitsAndBytesConfig
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from pyspark.sql import SparkSession
from langchain_community.vectorstores import ElasticsearchStore
from optimum.onnxruntime import ORTModelForCausalLM
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration,pipeline

import os

from utils.data import get_data
from utils.model import get_model_pipeline,get_final_chain,llm_query

spark = SparkSession.builder.appName("datareader").getOrCreate()

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant = False
)

#saved  model after static quantization and converting it to onnx
model_name = ""
config = AutoConfig.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

os.makedirs("tmp/trt_cache_gpt2_example", exist_ok=True)
provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "tmp/trt_cache_gpt2_example"
}

model = ORTModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    provider="TensorrtExecutionProvider",
    provider_options=provider_options,
)
embd_func = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

docs = get_data("dataset/yelp.json",spark)

vectorstore = ElasticsearchStore.from_documents(
    docs,
    embd_func,
    es_url="http://localhost:9200",
    index_name="yelp_index",
)

vectorstore.client.indices.refresh(index="yelp_index")

sane_pipeline = get_model_pipeline(model,tokenizer,temperature=0.0)
insane_pipeline=get_model_pipeline(model,tokenizer,temperature=0.2)


memory = ConversationBufferMemory(
    return_messages=True,output_key='answer',input_key='question'
) 

chain = get_final_chain(sane_pipeline,insane_pipeline,memory,vectorstore)
sampling_rate = 44100
duration = 3

input_audio = sd.rec(int(duration*sampling_rate),sampling_rate=sampling_rate,channels=2)
sd.wait()

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

encoded_audio = processor(input_audio,sampling_rate=sampling_rate).input_featurers
res_text = model.generate(encoded_audio)
input_string = processor.batch_decode(res_text,skip_special_tokens=True)

res_text = llm_query(input_string,chain,memory)

pipe = pipeline("text-to-speech",model='suno/bark-small')
output = pipe(res_text)

#output has audio and sampling rate