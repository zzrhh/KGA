from neo4j import GraphDatabase
import json
import pandas as pd
import ast
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def insert_member_paths(neo4j_uri, neo4j_user, neo4j_password, jsonl_path, database="neo4j"):
    """
    从一个 JSONL 文件中导入知识图谱路径中的三元组到 Neo4j 数据库。

    参数:
    - neo4j_uri: Neo4j 连接 URI（例如 "bolt://localhost:7688"）
    - neo4j_user: 数据库用户名
    - neo4j_password: 数据库密码
    - jsonl_path: 含路径信息的 JSONL 文件路径
    - database: 使用的 Neo4j 数据库名，默认为 "neo4j"
    """

    def create_triplet(tx, head, relation, tail):
        query = f"""
        MERGE (h:Entity {{name: $head}})
        MERGE (t:Entity {{name: $tail}})
        MERGE (h)-[:`{relation}`]->(t)
        """
        tx.run(query, head=head, tail=tail)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()

    count = 0
    with driver.session(database=database) as session:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                item = json.loads(line)
                print(f"[第{line_num + 1}行] 读取内容: {item}")
                path = item.get("path", [])
                if not path:
                    print(f"⚠️ 第{line_num + 1}行无有效 path 字段")
                    continue
                for h, r, t in path:
                    print(f"Inserting: ({h}) -[{r}]-> ({t})")
                    session.write_transaction(create_triplet, h, r, t)
                    count += 1

    driver.close()
    print(f"✅ 共导入 {count} 条三元组")


def preprocess_features(df, feature_col="feature"):
    df[feature_col] = df[feature_col].apply(ast.literal_eval)
    feature_df = pd.DataFrame(df[feature_col].tolist(), columns=[f"f_{i}" for i in range(len(df[feature_col].iloc[0]))])
    return pd.concat([feature_df, df["label"]], axis=1)

def split_data(pos_df, neg_df, test_ratio=0.9):
    # 不匹配行数，直接拼接切分
    pos_split = int(test_ratio * len(pos_df))
    neg_split = int(test_ratio * len(neg_df))

    pos_test, pos_train = pos_df.iloc[:pos_split], pos_df.iloc[pos_split:]
    neg_test, neg_train = neg_df.iloc[:neg_split], neg_df.iloc[neg_split:]

    train_df = pd.concat([pos_train, neg_train], ignore_index=True).reset_index(drop=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True).reset_index(drop=True)
    return train_df, test_df

def train_model(train_df, model_output_path, feature_col="feature"):
    train_processed = preprocess_features(train_df, feature_col)
    train_data = TabularDataset(train_processed)
    predictor = TabularPredictor(label="label", path=model_output_path).fit(train_data=train_data)
    print(f"✅ 模型已训练完成，保存在: {model_output_path}")
    return predictor

def evaluate_model(predictor, test_df, feature_col="feature"):
    test_processed = preprocess_features(test_df, feature_col)
    X_test = test_processed.drop(columns=["label"])
    y_test = test_processed["label"]

    y_pred = predictor.predict(X_test)
    y_prob = predictor.predict_proba(X_test)[1]

    print("✅ Evaluation Results:")
    print(f"AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"ACC:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"PRE:  {precision_score(y_test, y_pred):.4f}")
    print(f"REC:  {recall_score(y_test, y_pred):.4f}")
    print(f"F1:   {f1_score(y_test, y_pred):.4f}")
