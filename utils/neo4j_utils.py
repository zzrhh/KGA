from neo4j import GraphDatabase, AsyncGraphDatabase
from typing import List, Dict, Optional

class Neo4jQueryRunner:
    def __init__(self, uri: str, username: str, password: str):
        """
        初始化同步和异步 Neo4j 驱动
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.async_driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """
        关闭同步连接
        """
        if self.driver:
            self.driver.close()

    async def aclose(self):
        """
        关闭异步连接
        """
        if self.async_driver:
            await self.async_driver.close()

    def run_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        同步执行 Cypher 查询

        Args:
            cypher_query (str): 要执行的 Cypher 查询
            parameters (dict, optional): 可选参数

        Returns:
            List[Dict]: 查询结果
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]

    async def run_query_async(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        异步执行 Cypher 查询

        Args:
            cypher_query (str): 要执行的 Cypher 查询
            parameters (dict, optional): 可选参数

        Returns:
            List[Dict]: 查询结果
        """
        async with self.async_driver.session() as session:
            result = await session.run(cypher_query, parameters or {})
            records = await result.data()  # ✅ 正确方法
            return records



# 示例使用（你可以在实际代码中替换掉以下内容）
if __name__ == "__main__":
    runner = Neo4jQueryRunner("bolt://localhost:7688", "neo4j", "wu20020921")

    cypher = """
    MATCH (film:Entity)
    WHERE toLower(film.name) CONTAINS "club sandwich"
    WITH film
    MATCH (film)-[:directed_by]->(director)-[:born_in]->(place)
    RETURN place.name
    """

    results = runner.run_query(cypher)
    for row in results:
        print(row)

    runner.close()