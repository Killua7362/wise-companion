import transformers
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from operator import itemgetter
from langchain.memory.buffer import get_buffer_string
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from utils.templates import final_question_prompt,initial_question_prompt
from utils.data import combine_docs

def get_model_pipeline(model,tokenizer,temperature=0.0):
    sane_llm = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=temperature,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1000,
        do_sample=True if temperature != 0.0 else False,
    )
    return HuggingFacePipeline(pipeline=sane_llm)

def get_retriever(sane_pipeline,vectorstore):
    
    metadata_field_info = [
        AttributeInfo(
            name = 'fulladdress',
            description="This is the full address of business",
            type="string"
        ),
        AttributeInfo(
            name = 'categories',
            description="The string contains list of categories seperated by comma",
            type="string"
        ),
        AttributeInfo(
            name = 'stars',
            description="This is the review of the business",
            type="float"
        ),
    ]

    content_summary = "Name of the business"

    examples = [
        (
            "I want a hotel in the Balkans with a king sized bed and a hot tub. review is below 3",
            {
                "query": "",
                "filter": 'and(in("fulladdress", ["Balkans"]),and(in("categories", ["Mexican Resturant"]), lte("stars", 3))',
            },
        ),
        (
            "A room with breakfast with review above 3, at a Hilton",
            {
                "query": "",
                "filter":  'and(in("fulladdress", ["Hilton"]), gte("stars", 3))',
            },
        ),
    ]

    retrieval_chain = load_query_constructor_runnable(
        document_contents=content_summary,
        attribute_info=metadata_field_info,
        fix_invalid=True,
        enable_limit=True,
        llm=sane_pipeline,
        examples=examples
    )

    retriever = SelfQueryRetriever(
        query_constructor=retrieval_chain,
        vectorstore=vectorstore,
        verbose=False,
    )
    
    return retriever

def get_final_chain(sane_pipeline,insane_pipeline,memory,vectorstore):
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
    )
    
    retriever = get_retriever(sane_pipeline,vectorstore)

    initial_question = {
        "query":{
            "question":lambda x:x["question"],
            "chat_history":lambda x:get_buffer_string(x["chat_history"])
        }
        | initial_question_prompt
        | sane_pipeline
    }

    retrieved_document = {
        "docs":lambda x:retriever.invoke({"query":x['query']}),
        "question":lambda x:x['query']
    }

    final_inputs = {
        "context":lambda x:combine_docs(x["docs"]),
        "question":itemgetter("question")
    }

    answer = {
        "answer":final_inputs | final_question_prompt | insane_pipeline,
        "question":itemgetter('question'),
        "context":final_inputs["context"]
    }

    chain = loaded_memory | initial_question | retrieved_document | answer
    return chain


def llm_query(string,chain,memory):
    inputs = {"question":string}
    result = chain.invoke(inputs)

    memory.save_context(inputs, {"answer": result["answer"]})
    return result["answer"]