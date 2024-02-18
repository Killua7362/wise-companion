from langchain.prompts import PromptTemplate

template = """
[INST] 
Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language, 
that can be used to query a database. This query will be used to retrieve documents with additional context.

Let me share a couple examples.

If you do not see any chat history, you MUST return the "Follow Up Input" as is:
```
Chat History:
Follow Up Input: What is the nearest restaurant in LA?
Standalone Question:
What is the nearest restaurant in LA?
```

If this is the second question onwards, you should properly rephrase the question like this:
```
Chat History:
Human: What is the nearest restaurant in LA?
AI: 
The Forte is the nearest restaurant in LA.
Follow Up Input: What is it's rating?
Standalone Question:
What is the Forte's rating?
```

Now, with those examples, here is the actual chat history and input question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
[your response here]
[/INST] 
"""

initial_question_prompt = PromptTemplate.from_template(template=template)
template = """
[INST] 
While answering take below context into consideration:
{context}

Question: {question}
[/INST] 
"""
final_question_prompt = PromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="""
            [INST] 
            {page_content} is the name of the business
            {categories} are the categories mentioned for {page_content}
            {page_content} has {stars} star rating
            and it's full address is {fulladdress}
            [/INST] 
            """)
