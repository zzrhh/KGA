import json
from openai import AsyncOpenAI
import jsonlines
from tqdm.asyncio import tqdm as tqdm_asyncio
import asyncio
client = AsyncOpenAI(api_key="sk-Fne9clk28DYtrpOZq3ALtuUXR5kVW9vOt4ObyBHxolo7hfCO",base_url="https://xdaicn.top/v1")

def generate_cypher_prompt_OGBN(question):
    CYPHER_GENERATION_TEMPLATE = f"""
    Task: Generate a Cypher statement to extract all valid reasoning paths that answer a question over a knowledge graph.

    Instructions:
    --All reasoning paths MUST be strictly linear and unidirectional (e.g., A → B → C → D).
    --NEVER use left-pointing arrows (<-) in any MATCH clause. All relationships must point forward (→).
    --Use only the relationship types and properties provided in the schema.
    --All nodes are of type :Entity and have a property name.
    --Split the question entity name into multiple keywords (e.g., "author_533229" → "author", "533229").
    --Use toLower(n.name) CONTAINS each keyword, and combine with AND to match possible starting entities.
    --Try each matching entity as the starting point, and return only the valid paths that successfully follow the reasoning chain.
    --Use only one-hop relationships per MATCH clause.
    --Do not invent properties or relationships.
    --Do not include explanations or comments—only output a Cypher statement.
    --Return the full reasoning path using RETURN path, and ignore paths that do not fully resolve.

    Schema:
    Node type:
    (:Entity)
    Properties:
    - name: STRING

    Relationships:
    (:Entity)-[:cites]->(:Entity)
    (:Entity)-[:writes]->(:Entity)
    (:Entity)-[:has_topic]->(:Entity)

    ---

    Example 1:
    Question:
    What paper is cited by the paper written by author_963294?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "author" AND e1.name CONTAINS "963294"
    WITH e1
    MATCH path = (e1)-[:writes]->(e2:Entity)-[:cites]->(e3:Entity)
    RETURN path

    ---

    Example 2:
    Question:
    What is the field of study of the paper written by author_96344?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "author" AND e1.name CONTAINS "96344"
    WITH e1
    MATCH path = (e1)-[:writes]->(e2:Entity)-[:has_topic]->(e3:Entity)
    RETURN path

    ---

    Example 3:
    Question:
    What is the field of study of the paper cited by a paper written by author_626444?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "author" AND e1.name CONTAINS "626444"
    WITH e1
    MATCH path = (e1)-[:writes]->(e2:Entity)-[:cites]->(e3:Entity)-[:has_topic]->(e4:Entity)
    RETURN path

    ---

    Example 4:
    Question:
    What is the field of study of the paper cited by a paper that is cited by paper_725428?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "paper" AND e1.name CONTAINS "725428"
    WITH e1
    MATCH path = (e1)-[:cites]->(e2:Entity)-[:cites]->(e3:Entity)-[:has_topic]->(e4:Entity)
    RETURN path

    ---

    Question:
    {question}

    Cypher:


    """
    return CYPHER_GENERATION_TEMPLATE


def generate_cypher_prompt_meta_QA(question):
    CYPHER_GENERATION_TEMPLATE = f"""
    Task: Generate a Cypher query to extract valid reasoning paths from a Neo4j knowledge graph in response to a given question.

    Instructions:
    --All reasoning paths MUST be strictly linear and unidirectional (e.g., A → B → C → D).
    --NEVER use left-pointing arrows (<-) in any MATCH clause. All relationships must point forward (→).
    --Use only the relationship types and properties provided in the schema.
    --All nodes are of type :Entity and have a property name.
    --Split the question entity name into multiple keywords (e.g., "Jackie Chan" → "jackie", "chan").
    --Use toLower(n.name) CONTAINS each keyword, and combine with AND to match possible starting entities.
    --Try each matching entity as the starting point, and return only the valid paths that successfully follow the reasoning chain.
    --Use only one-hop relationships per MATCH clause.
    --Do not invent properties or relationships.
    --Do not include explanations or comments—only output a Cypher statement.
    --Return the full reasoning path using RETURN path, and ignore paths that do not fully resolve.

    Schema:
    Node type:
    (:Entity)
    Properties:
    - name: STRING

    Relationships:
    (:Entity)-[:`was released in year`]->(:Entity)
    (:Entity)-[:`is release year of`]->(:Entity)
    (:Entity)-[:`was directed by`]->(:Entity)
    (:Entity)-[:`directed`]->(:Entity)
    (:Entity)-[:`wrote`]->(:Entity)
    (:Entity)-[:`was written by`]->(:Entity)
    (:Entity)-[:`has starred actors`]->(:Entity)
    (:Entity)-[:`starred in`]->(:Entity)
    (:Entity)-[:`has genre`]->(:Entity)
    (:Entity)-[:`is genre of`]->(:Entity)
    (:Entity)-[:`has tag`]->(:Entity)
    (:Entity)-[:`is tag of`]->(:Entity)
    (:Entity)-[:`was in language`]->(:Entity)
    (:Entity)-[:`is language spoken in`]->(:Entity)
    (:Entity)-[:`has rating`]->(:Entity)
    (:Entity)-[:`is rating of`]->(:Entity)

    ---

    Example 1:
    Question:
    What are the languages of the films directed by [Joel Zwick]?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "joel" AND toLower(e1.name) CONTAINS "zwick"
    WITH e1
    MATCH path = (e1)-[:`directed`]->(film:Entity)-[:`was in language`]->(lang:Entity)
    RETURN path

    ---

    Example 2:
    Question:
    Who directed the movies written by [Jackie Chan]?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "jackie" AND toLower(e1.name) CONTAINS "chan"
    WITH e1
    MATCH path = (e1)-[:`wrote`]->(film:Entity)-[:`was directed by`]->(dir:Entity)
    RETURN path

    ---

    Example 3:
    Question:
    What are the genres of the movies directed by the writers of [Finding Forrester]?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "finding" AND toLower(e1.name) CONTAINS "forrester"
    WITH e1
    MATCH path = (e1)-[:`was written by`]->(writer:Entity)-[:`wrote`]->(film:Entity)-[:`has genre`]->(genre:Entity)
    RETURN path

    ---

    Example 4:
    Question:
    Who starred in the films directed by the director of [Iron Man 2]?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "iron" AND toLower(e1.name) CONTAINS "man" AND toLower(e1.name) CONTAINS "2"
    WITH e1
    MATCH path = (e1)-[:`was directed by`]->(dir:Entity)-[:`directed`]->(film:Entity)-[:`has starred actors`]->(actor:Entity)
    RETURN path

    ---

    Question:
    {question}

    Cypher:
        """
    return CYPHER_GENERATION_TEMPLATE




def generate_cypher_prompt_2wiki_webquestions(question):
    CYPHER_GENERATION_TEMPLATE = f"""
    Task: Generate a Cypher statement to extract all valid reasoning paths that answer a question over a knowledge graph.

    Instructions:
    --All reasoning paths MUST be strictly linear and unidirectional (e.g., A → B → C → D).
    --NEVER use left-pointing arrows (<-) in any MATCH clause. All relationships must point forward (→).
    --Use only the relationship types and properties provided in the schema below.
    --All nodes are of type :Entity and have a property name.
    --Split the question entity name into multiple keywords (e.g., "barack obama" → "barack", "obama").
    --Use toLower(n.name) CONTAINS each keyword, and combine with AND to match possible starting entities.
    --Try each matching entity as the starting point, and return only the valid paths that successfully follow the reasoning chain.
    --Use only one-hop relationships per MATCH clause.
    --Enclose all relationship types in backticks, e.g., [:`directed_by`].
    --Return the full reasoning path using RETURN path, and ignore paths that do not fully resolve.
    --Do not include explanations or comments—only output a valid Cypher query.

    Schema:
    Node type:
    (:Entity)
    Properties:
    - name: STRING

    Relationships:
    (:Entity)-[:`has_father`]->(:Entity)
    (:Entity)-[:`has_mother`]->(:Entity)
    (:Entity)-[:`has_child`]->(:Entity)
    (:Entity)-[:`sibling`]->(:Entity)
    (:Entity)-[:`married_to`]->(:Entity)
    (:Entity)-[:`date_of_birth`]->(:Entity)
    (:Entity)-[:`date_of_death`]->(:Entity)
    (:Entity)-[:`born_in`]->(:Entity)
    (:Entity)-[:`died_in`]->(:Entity)
    (:Entity)-[:`cause_of_death`]->(:Entity)
    (:Entity)-[:`directed_by`]->(:Entity)
    (:Entity)-[:`composed_by`]->(:Entity)
    (:Entity)-[:`performed_by`]->(:Entity)
    (:Entity)-[:`citizenship`]->(:Entity)
    (:Entity)-[:`founded_by`]->(:Entity)
    (:Entity)-[:`educated_at`]->(:Entity)
    (:Entity)-[:`published_by`]->(:Entity)
    (:Entity)-[:`employed_by`]->(:Entity)
    (:Entity)-[:`award_received`]->(:Entity)
    (:Entity)-[:`detained_at`]->(:Entity)
    (:Entity)-[:`edited_by`]->(:Entity)
    (:Entity)-[:`created_by`]->(:Entity)
    (:Entity)-[:`presented_by`]->(:Entity)
    (:Entity)-[:`inception_year`]->(:Entity)
    (:Entity)-[:`buried_in`]->(:Entity)
    (:Entity)-[:`located_in`]->(:Entity)
    (:Entity)-[:`student of`]->(:Entity)
    (:Entity)-[:`has part`]->(:Entity)
    (:Entity)-[:`doctoral advisor`]->(:Entity)
    (:Entity)-[:`manufacturer`]->(:Entity)

    ---

    Example 1:
    Question:
    What is the date of death of the father of John Smith?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "john" AND toLower(e1.name) CONTAINS "smith"
    WITH e1
    MATCH path = (e1)-[:`has_father`]->(father:Entity)-[:`date_of_death`]->(d:Entity)
    RETURN path

    ---

    Example 2:
    Question:
    Where was the person employed who directed Inception?

    Cypher:
    MATCH (e1:Entity)
    WHERE toLower(e1.name) CONTAINS "inception"
    WITH e1
    MATCH path = (e1)-[:`directed_by`]->(director:Entity)-[:`employed_by`]->(org:Entity)
    RETURN path

    ---

    Question:
    {question}

    Cypher:
    """
    return CYPHER_GENERATION_TEMPLATE

def generate_QA_prompt(question, cypher_answer):
    cypher_answer_str = json.dumps(cypher_answer, ensure_ascii=False)
    QA_PROMPT = """
    Task: Generate a Precise and Clear answer of the given question with provided context.

    You should provide your answer strictly follow the rules below:
    --Use the context to support your answer. The context contains the reasoning path regarding the question.
    --Answer the question based on the context precisely and briefly. 
    --If you can't infer the answer from the context, just say so.

    ---
    Answer the question precisely and concisely, using only one word or phrase—no introduction, explanation, prefix, or suffix.

    Question:
        {question}

    Context:
        {cypher_answer}

    Answer:
    """.format(
            question=question,
            cypher_answer=cypher_answer_str.replace("{", "{{").replace("}", "}}")
        )

    return QA_PROMPT
