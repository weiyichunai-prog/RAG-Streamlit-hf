"""
Streamlit + Groq API - 8種 RAG 策略 PDF 問答系統
需要安裝: pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn
執行方式: streamlit run LLM-RAG-Streamlit.py
"""

import streamlit as st
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="🤖 多策略 RAG PDF 問答系統",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自訂樣式 ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .strategy-card {
        background: #f9fafb;
        border-left: 4px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== RAG 核心類別 ====================
class MultiStrategyRAG:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    @st.cache_resource
    def _load_embedding_model(_self):
        return SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

    def load_pdf(self, pdf_file):
        """載入 PDF 檔案"""
        try:
            reader = PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            self.chunks = self._split_text(full_text, chunk_size=800, overlap=150)

            embedding_model = self._load_embedding_model()

            with st.spinner("⚙️ 正在生成嵌入向量..."):
                self.embeddings = embedding_model.encode(
                    self.chunks, convert_to_numpy=True, show_progress_bar=False
                )

            # 建立 FAISS 索引
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))

            # 建立 TF-IDF 索引
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            return True, f"✅ 成功載入！共 {len(reader.pages)} 頁，分割為 {len(self.chunks)} 個片段"

        except Exception as e:
            return False, f"❌ 載入失敗: {str(e)}"

    def _split_text(self, text, chunk_size, overlap):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunk = re.sub(r'\s+', ' ', text[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ==================== 8種 RAG 策略 ====================

    def strategy_1_basic_similarity(self, query, top_k=3):
        embedding_model = self._load_embedding_model()
        query_vector = embedding_model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def strategy_2_tfidf(self, query, top_k=3):
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices]

    def strategy_3_hybrid(self, query, top_k=3):
        embedding_model = self._load_embedding_model()
        query_vector = embedding_model.encode([query])
        distances, sem_indices = self.index.search(query_vector.astype('float32'), top_k * 2)

        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_tfidf.T).toarray().flatten()
        tfidf_indices = tfidf_scores.argsort()[-top_k * 2:][::-1]

        combined = list(set(sem_indices[0].tolist() + tfidf_indices.tolist()))
        return [self.chunks[idx] for idx in combined[:top_k]]

    def strategy_4_reranking(self, query, top_k=3):
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)
        reranked = []
        for chunk in candidates:
            prompt = f"問題：{query}\n\n文本：{chunk[:200]}...\n\n這段文本與問題的相關度(0-10)："
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                score_text = response.choices[0].message.content.strip()
                score = float(re.findall(r'\d+', score_text)[0]) if re.findall(r'\d+', score_text) else 0
                reranked.append((chunk, score))
            except:
                reranked.append((chunk, 0))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in reranked[:top_k]]

    def strategy_5_multi_query(self, query, top_k=3):
        expansion_prompt = f"將以下問題改寫成3個相關但不同角度的問題，用換行分隔：\n{query}"
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            queries = [query] + response.choices[0].message.content.strip().split('\n')[:3]
        except:
            queries = [query]

        all_chunks = []
        for q in queries:
            all_chunks.extend(self.strategy_1_basic_similarity(q, top_k=2))
        return list(dict.fromkeys(all_chunks))[:top_k]

    def strategy_6_contextual_compression(self, query, top_k=3):
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            compress_prompt = f"從以下文本中提取與問題「{query}」最相關的1-2句話：\n\n{chunk}"
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": compress_prompt}],
                    max_tokens=150,
                    temperature=0
                )
                compressed.append(response.choices[0].message.content.strip())
            except:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query, top_k=3):
        embedding_model = self._load_embedding_model()
        small_chunks = self._split_text(' '.join(self.chunks), chunk_size=300, overlap=50)
        small_embeddings = embedding_model.encode(small_chunks, convert_to_numpy=True, show_progress_bar=False)

        small_index = faiss.IndexFlatL2(small_embeddings.shape[1])
        small_index.add(small_embeddings.astype('float32'))

        query_vector = embedding_model.encode([query])
        distances, indices = small_index.search(query_vector.astype('float32'), top_k)

        results = []
        for idx in indices[0]:
            for big_chunk in self.chunks:
                if small_chunks[idx] in big_chunk:
                    results.append(big_chunk)
                    break
        return list(dict.fromkeys(results))[:top_k]

    def strategy_8_hypothetical_answer(self, query, top_k=3):
        embedding_model = self._load_embedding_model()
        hyde_prompt = f"請對以下問題給出一個假設性的答案（即使不確定）：\n{query}"
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            hypothetical_answer = response.choices[0].message.content
        except:
            hypothetical_answer = query

        query_vector = embedding_model.encode([hypothetical_answer])
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def generate_answer(self, query, strategy, top_k=3):
        if not self.chunks:
            return None, None, "❌ 請先上傳 PDF 檔案！"

        strategies = {
            "1. 基礎語意搜尋":      self.strategy_1_basic_similarity,
            "2. TF-IDF 關鍵詞":     self.strategy_2_tfidf,
            "3. 混合搜尋":          self.strategy_3_hybrid,
            "4. 重新排序":          self.strategy_4_reranking,
            "5. 多查詢擴展":        self.strategy_5_multi_query,
            "6. 上下文壓縮":        self.strategy_6_contextual_compression,
            "7. 父子文檔":          self.strategy_7_parent_child,
            "8. 假設性答案 (HyDE)": self.strategy_8_hypothetical_answer,
        }

        retrieval_func = strategies.get(strategy, self.strategy_1_basic_similarity)
        relevant_chunks = retrieval_func(query, top_k)
        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = f"""請根據以下上下文回答問題。如果上下文中沒有相關資訊，請說明無法回答。

上下文：
{context}

問題：{query}

請用繁體中文詳細回答："""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "你是專業的文件分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            return answer, relevant_chunks, None

        except Exception as e:
            return None, None, f"❌ 生成答案失敗: {str(e)}"


# ==================== Session State 初始化 ====================
API_KEY = "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"

if "rag" not in st.session_state:
    st.session_state.rag = MultiStrategyRAG(api_key=API_KEY)
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "load_message" not in st.session_state:
    st.session_state.load_message = ""


# ==================== 側邊欄 ====================
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    st.divider()

    # 上傳 PDF
    st.markdown("### 📤 步驟 1：上傳 PDF")
    uploaded_file = st.file_uploader("選擇 PDF 檔案", type=["pdf"], label_visibility="collapsed")

    if st.button("🚀 載入文件", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ 請先選擇 PDF 檔案")
        else:
            with st.spinner("📖 正在解析 PDF..."):
                ok, msg = st.session_state.rag.load_pdf(uploaded_file)
                st.session_state.pdf_loaded = ok
                st.session_state.load_message = msg

    if st.session_state.load_message:
        if st.session_state.pdf_loaded:
            st.success(st.session_state.load_message)
        else:
            st.error(st.session_state.load_message)

    st.divider()

    # 策略選擇
    st.markdown("### 🧠 步驟 2：選擇 RAG 策略")
    strategy = st.selectbox(
        "RAG 策略",
        options=[
            "1. 基礎語意搜尋",
            "2. TF-IDF 關鍵詞",
            "3. 混合搜尋",
            "4. 重新排序",
            "5. 多查詢擴展",
            "6. 上下文壓縮",
            "7. 父子文檔",
            "8. 假設性答案 (HyDE)",
        ],
        label_visibility="collapsed"
    )

    top_k = st.slider("檢索片段數量 (Top-K)", min_value=1, max_value=10, value=3, step=1)

    st.divider()

    # 策略說明
    st.markdown("### 📖 策略說明")
    strategy_descriptions = {
        "1. 基礎語意搜尋":      "使用向量相似度找出語意最接近的片段",
        "2. TF-IDF 關鍵詞":     "基於詞頻統計做關鍵詞比對",
        "3. 混合搜尋":          "結合語意搜尋與 TF-IDF 的優點",
        "4. 重新排序":          "先廣泛檢索，再用 LLM 重新評分排序",
        "5. 多查詢擴展":        "自動生成多個相關問題並彙整結果",
        "6. 上下文壓縮":        "從每個片段中提取最相關的句子",
        "7. 父子文檔":          "用小片段定位，回傳完整大片段上下文",
        "8. 假設性答案 (HyDE)": "先讓 LLM 生成假設答案，再以此搜尋",
    }
    for name, desc in strategy_descriptions.items():
        icon = "👉" if name == strategy else "▸"
        st.markdown(
            f'<div class="strategy-card"><b>{icon} {name}</b><br><span style="color:#6b7280">{desc}</span></div>',
            unsafe_allow_html=True
        )


# ==================== 主畫面 ====================
st.markdown('<div class="main-title">🤖 多策略 RAG PDF 問答系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">採用 8 種不同的 RAG 策略，為您的 PDF 文件提供智能問答服務</div>', unsafe_allow_html=True)

# 提示：PDF 未載入
if not st.session_state.pdf_loaded:
    st.info("👈 請先在左側側邊欄上傳並載入 PDF 文件，然後再開始提問。")

st.markdown("### 💬 步驟 3：提問")

# 範例問題快速鍵
EXAMPLE_QUESTIONS = [
    "這份文件的主要內容是什麼？",
    "文件中提到哪些重要概念？",
    "有哪些關鍵數據或統計資料？",
    "文件的結論是什麼？",
]

st.markdown("**💡 範例問題：**")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(example, use_container_width=True):
        st.session_state["question_input"] = example

# 問題輸入框
question = st.text_area(
    "輸入您的問題",
    value=st.session_state.get("question_input", ""),
    placeholder="例如：這份文件的主要內容是什麼？",
    height=100,
    label_visibility="collapsed",
)

ask_clicked = st.button("🔍 開始提問", type="primary", use_container_width=True, disabled=not st.session_state.pdf_loaded)

# ==================== 執行問答 ====================
if ask_clicked:
    if not question.strip():
        st.warning("⚠️ 請輸入問題後再提交！")
    else:
        with st.spinner(f"🤔 正在使用「{strategy}」策略搜尋並生成答案..."):
            answer, chunks, error = st.session_state.rag.generate_answer(question, strategy, top_k)

        if error:
            st.error(error)
        else:
            st.divider()
            st.markdown("### 💡 AI 回答")
            st.markdown(
                f'<div style="background:#f0f9ff;border-left:4px solid #0ea5e9;padding:1rem 1.25rem;border-radius:0 0.5rem 0.5rem 0;line-height:1.75">{answer}</div>',
                unsafe_allow_html=True
            )

            st.divider()
            st.markdown(f"📚 **使用策略**：{strategy} ｜ 📄 **檢索片段數**：{len(chunks)}")

            with st.expander("📂 查看檢索到的文本片段", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**片段 {i}**")
                    st.text_area(
                        label=f"chunk_{i}",
                        value=chunk,
                        height=120,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    if i < len(chunks):
                        st.divider()
