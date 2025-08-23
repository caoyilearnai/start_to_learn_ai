# 导入所需的库
import numpy as np
import faiss
import pickle
import os
# 需要安装 openai: pip install openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ----------------------------
# 一、配置参数
# ----------------------------

VECTOR_DB_DIR = "vector_db"
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(VECTOR_DB_DIR, "documents.pkl")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# 通义千问模型名称 (请根据你在百炼平台选择的模型更改)
GENERATION_MODEL_NAME = 'qwen-plus' # 或者 'qwen-turbo', 'qwen-max', 'qwen-long'

# 设置 OpenAI 风格的客户端以调用阿里云百炼平台
# 请将 'YOUR_DASHSCOPE_API_KEY' 替换为你在阿里云百炼平台获取的实际 API Key
# 或者设置环境变量 DASHSCOPE_API_KEY
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key or api_key == 'YOUR_DASHSCOPE_API_KEY':
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或在代码中配置有效的 API Key")

# 配置 OpenAI 客户端使用阿里云百炼服务
llm_client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ----------------------------
# 二、加载预训练模型 (仅嵌入模型)
# ----------------------------

print("🔧 正在加载嵌入模型...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ----------------------------
# 三、定义本地向量数据库类
# ----------------------------

class SimpleVectorDB:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_loaded = False

    def create_and_save(self, documents):
        print("🧠 正在为文档生成向量表示（嵌入）...")
        doc_embeddings = embedding_model.encode(documents)
        doc_embeddings = np.array(doc_embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(doc_embeddings)

        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)

        with open(DOCS_PATH, 'wb') as f:
            pickle.dump(documents, f)

        self.documents = documents
        self.is_loaded = True

        print(f"✅ 成功创建并向量数据库保存到:")
        print(f"   - 索引文件: {INDEX_PATH}")
        print(f"   - 文档文件: {DOCS_PATH}")

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(
                f"❌ 找不到数据库文件！\n"
                f"请先运行 create_and_save() 创建数据库。\n"
                f"需要的文件:\n"
                f"  {INDEX_PATH}\n"
                f"  {DOCS_PATH}"
            )

        print("📂 正在从磁盘加载向量数据库...")
        self.index = faiss.read_index(INDEX_PATH)

        with open(DOCS_PATH, 'rb') as f:
            self.documents = pickle.load(f)

        self.is_loaded = True
        print("✅ 向量数据库加载成功！")
        print(f"📊 共加载 {self.index.ntotal} 个文档向量")
        print(f"📘 共 {len(self.documents)} 条原始文本")

    def search(self, query, top_k=1):
        if not self.is_loaded:
            raise RuntimeError("❌ 数据库还未加载，请先调用 load() 或 create_and_save()")

        print(f"🔍 正在检索与 '{query}' 最相关的文档...")
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            doc_idx = indices[0][i]
            if doc_idx == -1:
                continue
            text = self.documents[doc_idx]
            score = float(distances[0][i])
            results.append({'text': text, 'score': score})

        return results

# ----------------------------
# 四、RAG 主函数 (使用通义千问 - OpenAI API 风格)
# ----------------------------

def rag_query(db, question, top_k=1, model_name=GENERATION_MODEL_NAME):
    """
    RAG 核心流程：检索相关文档 + 调用通义千问生成回答 (OpenAI API 风格)
    :param db: 向量数据库对象
    :param question: 用户的问题
    :param top_k: 检索前 k 个相关文档
    :param model_name: 通义千问模型名称
    :return: 包含问题、上下文、回答的字典
    """
    # 1. 从数据库中检索与问题最相关的文档
    search_results = db.search(question, top_k=top_k)

    # 2. 提取检索到的文档内容，拼成“上下文”
    retrieved_docs = [res['text'] for res in search_results]
    context = "\n".join(retrieved_docs) # 用换行符连接

    # 3. 构造提示词（Prompt）：告诉模型根据上下文回答问题
    messages = [
        {"role": "system", "content": "你是一个 helpful 的 AI 助手。请根据提供的信息回答问题。如果信息不足以回答，请说明原因，不要编造答案。"},
        {"role": "user", "content": f"信息：\n{context}\n\n问题：{question}\n\n回答："}
    ]

    # 4. 调用通义千问API生成回答 (OpenAI API 风格)
    print("🤖 正在调用通义千问生成回答...")
    try:
        completion = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            # 可以根据需要调整参数
            max_tokens=200,
            temperature=0.7,
            top_p=0.8
        )
        # 提取生成的文本
        answer = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ 调用通义千问时出现异常: {e}")
        answer = f"抱歉，调用模型时出现异常: {e}"

    # 返回完整结果
    return {
        "question": question,
        "retrieved_context": context,
        "messages": messages, # 返回用于调试的 messages
        "answer": answer
    }


# ----------------------------
# 五、主程序入口
# ----------------------------

if __name__ == "__main__":
    documents = [
        "Python 是一种高级编程语言，广泛用于 Web 开发、数据科学和人工智能。",
        "RAG 是 Retrieval-Augmented Generation 的缩写，结合了信息检索和文本生成。",
        "FAISS 是 Facebook 开源的向量相似性搜索库，支持高效检索。",
        "Sentence Transformers 可以将句子转换为高质量的向量表示。",
        "通义千问是由阿里云开发的超大规模语言模型，能够回答问题、创作文字。"
    ]

    db = SimpleVectorDB(dimension=384)

    # 检查数据库是否已存在，如果不存在则创建
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("未找到现有数据库，正在创建...")
        db.create_and_save(documents)
    else:
        print("找到现有数据库，正在加载...")
        db.load()

    # 简单的交互式问答循环
    print("\n--- 欢迎使用基于通义千问(OpenAI API)的简易RAG问答系统 ---")
    print("输入 '退出' 或 'quit' 结束程序。")
    while True:
        user_input = input("\n请输入您的问题: ").strip()
        if user_input.lower() in ['退出', 'quit']:
            print("再见！")
            break
        if user_input:
            try:
                result = rag_query(db, user_input, top_k=3)
                print(f"\n❓ 问题: {result['question']}")
                print(f"📄 检索到的信息: {result['retrieved_context']}")
                print(f"💡 通义千问回答: {result['answer']}")
            except Exception as e:
                print(f"处理问题时出错: {e}")
        else:
            print("请输入一个有效问题。")