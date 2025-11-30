import re
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, LayerNormalization, \
    Multiply, Add, Activation, Conv1D, GlobalMaxPooling1D, Reshape, MultiHeadAttention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import BinaryCrossentropy  # 修正的导入语句
import tensorflow as tf
import jieba
import os
from transformers import BertTokenizer, TFBertModel
import time
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.initializers import Orthogonal

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 设置随机种子确保可复现性
tf.random.set_seed(42)
np.random.seed(42)

# 优化分词效率 - 添加缓存机制
jieba.enable_parallel(8)  # 启用并行分词
jieba_cache = {}
jieba.setLogLevel('ERROR')  # 减少日志干扰


# 自定义F1分数指标
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)
        self.f1 = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        p = self.precision.result()
        r = self.recall.result()
        self.f1.assign(2 * ((p * r) / (p + r + tf.keras.backend.epsilon())))

    def result(self):
        return self.f1

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        self.f1.assign(0.0)


def efficient_preprocess(text, stopwords):
    """高效文本预处理函数 - 修复数字和标点处理"""
    if not isinstance(text, str) or pd.isna(text):
        return []

    if text in jieba_cache:
        return jieba_cache[text]

    # 修复1: 使用正则正确替换数字
    clean_text = re.sub(r'\d+', 'NUM', text)  # 确保数字被替换

    # 修复2: 完全保留标点符号
    words = [word for word in jieba.lcut(clean_text)
             if word.strip()]  # 不再过滤标点

    jieba_cache[text] = words
    return words


# 1. 数据加载与预处理 - 使用更高效的数据处理
def load_and_preprocess_data():
    """高效加载和预处理数据"""
    print("加载数据...")
    try:
        # 优化数据加载 - 更健壮的数据处理方式
        # 先以字符串类型读取所有数据
        data = pd.read_csv('data.csv', header=None,
                           encoding='gbk',
                           names=['text', 'label'],
                           usecols=[0, 1],
                           dtype=str,  # 先全部读取为字符串
                           engine='c',
                           on_bad_lines='skip')

        print(f"初步加载数据大小: {len(data)}")

        # 清理数据 - 确保文本和标签列存在
        data = data.dropna(subset=['text', 'label']).copy()

        # 转换标签列为整数类型
        # 1. 先移除可能的空白字符
        data['label'] = data['label'].str.strip()

        # 2. 尝试将标签转换为数值
        data['label'] = pd.to_numeric(data['label'], errors='coerce')

        # 3. 删除无法转换的标签行
        data = data.dropna(subset=['label']).copy()

        # 4. 转换为int8类型
        data['label'] = data['label'].astype('int8')

        # 处理文本列
        data['text'] = data['text'].str.strip()  # 去除两端空白
        print(f"有效数据大小: {len(data)}")

    except FileNotFoundError:
        print("错误: 找不到data.csv文件")
        return None, None, None
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None

    # 加载停用词 - 使用更高效的集合操作
    stopwords = set()
    try:
        with open('stopwords.txt', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print("警告: 未找到stopwords.txt文件，将使用空停用词表")

    # 划分数据集
    train_data, test_data = train_test_split(
        data,
        test_size=0.15,
        stratify=data['label'],
        random_state=42
    )

    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

    # 使用并行处理加速文本预处理
    print("并行预处理文本...")
    with ThreadPoolExecutor(max_workers=8) as executor:  # 增加工作线程数
        train_data['processed'] = list(executor.map(
            lambda x: efficient_preprocess(x, stopwords),
            train_data['text']
        ))
        test_data['processed'] = list(executor.map(
            lambda x: efficient_preprocess(x, stopwords),
            test_data['text']
        ))

    # 过滤空文本
    train_data = train_data[train_data['processed'].str.len() > 0]
    test_data = test_data[test_data['processed'].str.len() > 0]

    # 添加：过滤短文本
    train_data = train_data[train_data['processed'].apply(len) >= 3]
    test_data = test_data[test_data['processed'].apply(len) >= 3]

    return train_data, test_data, stopwords
# 加载情感词典 - 添加缓存机制
def load_emotion_dicts(vocab):
    """加载情感词典并优化存储 - 新增公安信访词典"""
    # 使用缓存避免重复加载
    if not hasattr(load_emotion_dicts, 'cache'):
        load_emotion_dicts.cache = {}

    cache_key = tuple(sorted(vocab))
    if cache_key in load_emotion_dicts.cache:
        return load_emotion_dicts.cache[cache_key]

    # 初始化词典
    positive_words = set()
    negative_words = set()
    emotion_intensity = {}  # 情感词强度词典

    # 1. 加载普通情感词典
    try:
        with open('NTUSD+my_pos.txt', encoding='utf-8') as f:
            words = set(line.strip() for line in f if line.strip())
            # 为积极词添加强度 (2.0-3.0)
            for word in words:
                emotion_intensity[word] = 2.0 + np.random.rand()  # 2.0-3.0
            positive_words |= words
    except FileNotFoundError:
        print("警告: 未找到普通积极情感词典文件")

    try:
        with open('NTUSD+my_neg.txt', encoding='utf-8') as f:
            words = set(line.strip() for line in f if line.strip())
            # 为消极词添加强度 (2.0-3.0)
            for word in words:
                emotion_intensity[word] = 2.0 + np.random.rand()  # 2.0-3.0
            negative_words |= words
    except FileNotFoundError:
        print("警告: 未找到普通消极情感词典文件")

    # 2. 加载公安信访情感词典 - 赋予更高强度
    try:
        with open('police_pos.txt', encoding='utf-8') as f:
            words = set(line.strip() for line in f if line.strip())
            # 为公安信访积极词赋予更高强度 (4.0-5.0)
            for word in words:
                emotion_intensity[word] = 4.0 + np.random.rand()  # 4.0-5.0
            positive_words |= words
        print(f"加载公安信访积极词典: {len(words)}个词")
    except FileNotFoundError:
        print("警告: 未找到公安信访积极极词典文件")

    try:
        with open('police_neg.txt', encoding='utf-8') as f:
            words = set(line.strip() for line in f if line.strip())
            # 为公安信访消极词赋予更高强度 (4.0-5.0)
            for word in words:
                emotion_intensity[word] = 4.0 + np.random.rand()  # 4.0-5.0
            negative_words |= words
        print(f"加载公安信访消极词典: {len(words)}个词")
    except FileNotFoundError:
        print("警告: 未找到公安信访消极词典文件")

    # 保留在词汇表中的情感词
    positive_words = positive_words & vocab
    negative_words = negative_words & vocab

    result = (positive_words, negative_words, emotion_intensity)
    load_emotion_dicts.cache[cache_key] = result
    return result


# 增强情感特征提取
def extract_emotional_features(words, positive_words, negative_words, emotion_intensity):
    """提取关键情感特征 - 增强版（新增公安信访特征）"""
    if not words:
        return np.zeros(14, dtype=np.float32)  # 特征维度14

    word_count = len(words)

    # 1. 情感词计数与强度计算
    pos_count = 0
    neg_count = 0
    emotion_scores = []
    emotion_positions = []

    # 遍历所有词，计算情感特征
    for i, word in enumerate(words):
        if word in positive_words:
            pos_count += 1
            intensity = emotion_intensity.get(word, 1.0)
            emotion_scores.append(intensity)
            # 位置权重: 文本开头和结尾的情感词更重要
            position_weight = 1.0 + 0.5 * (1 - abs(2 * i / word_count - 1))  # U型权重
            emotion_positions.append(position_weight)
        elif word in negative_words:
            neg_count += 1
            intensity = -emotion_intensity.get(word, 1.0)  # 消极词为负值
            emotion_scores.append(intensity)
            position_weight = 1.0 + 0.5 * (1 - abs(2 * i / word_count - 1))  # U型权重
            emotion_positions.append(position_weight)

    # 2. 高级情感特征计算
    sentiment_sum = sum(emotion_scores) if emotion_scores else 0.0
    sentiment_avg = sentiment_sum / max(len(emotion_scores), 1) if emotion_scores else 0.0

    # 情感波动: 情感变化的方差
    emotion_variance = np.var(emotion_scores) if len(emotion_scores) > 1 else 0.0

    # 情感强度变化: 情感强度变化趋势
    if len(emotion_scores) > 1:
        diff = [emotion_scores[i + 1] - emotion_scores[i] for i in range(len(emotion_scores) - 1)]
        intensity_change = sum(diff) / len(diff)
    else:
        intensity_change = 0.0

    # 修复标点计数 - 确保正确统计
    exclamation = words.count('!') + words.count('！')
    question = words.count('?') + words.count('？')

    # 新增: 否定词检测
    negation_words = {'不', '没', '无', '非', '未', '莫', '勿', '毋'}
    negation_count = sum(1 for word in words if word in negation_words)

    # 4. 位置加权情感得分
    weighted_sentiment = sum(s * p for s, p in zip(emotion_scores, emotion_positions)) / max(sum(emotion_positions), 1)

    # 新增: 情感词位置分布特征
    pos_positions = [i / len(words) for i, word in enumerate(words) if word in positive_words]
    neg_positions = [i / len(words) for i, word in enumerate(words) if word in negative_words]

    pos_position_mean = np.mean(pos_positions) if pos_positions else 0.0
    neg_position_mean = np.mean(neg_positions) if neg_positions else 0.0
    pos_position_var = np.var(pos_positions) if len(pos_positions) > 1 else 0.0
    neg_position_var = np.var(neg_positions) if len(neg_positions) > 1 else 0.0

    # 新增: 情感词密度特征
    emotion_density = len(emotion_scores) / max(len(words), 1)

    # 返回增强特征向量
    return np.array([
        pos_count,
        neg_count,
        sentiment_avg,
        weighted_sentiment,
        emotion_variance,
        intensity_change,
        sentiment_sum / max(word_count, 1),
        exclamation,
        question,
        negation_count,
        pos_position_mean,  # 新增积极词位置均值
        neg_position_mean,  # 新增消极词位置均值
        pos_position_var,  # 新增积极词位置方差
        emotion_density  # 新增情感词密度特征
    ], dtype=np.float32)


def build_enhanced_bert_textcnn_model(max_len, emo_dim, dropout_rate=0.2):
    """构建情感增强型BERT+TextCNN模型 - 公安信访优化版"""
    # 导入所需的卷积层
    from tensorflow.keras.layers import Conv1D, DepthwiseConv1D

    # 文本输入
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    # 情感特征输入
    emotion_input = Input(shape=(emo_dim,), name='emotion_input', dtype=tf.float32)

    try:
        print("加载优化的BERT模型...")
        model_path = "/root/autodl-fs/part2/gai/bert-base-chinese"

        if os.path.exists(model_path):
            bert_model = TFBertModel.from_pretrained(model_path)  # 删除add_pooling_layer参数
        else:
            bert_model = TFBertModel.from_pretrained("bert-base-chinese")  # 删除add_pooling_layer参数

        # 获取BERT输出
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        print("BERT模型加载成功，强制使用CPU模式处理卷积层")

        # ===== 修复卷积层问题 =====
        with tf.device('/device:CPU:0'):  # 明确指定CPU设备
            sequence_output = tf.cast(sequence_output, tf.float32)

            # 简化Conv1D参数，移除可能引发问题的正则化器
            conv_blocks = []
            filter_sizes = [2, 3, 4, 5]
            num_filters = 64

            for size in filter_sizes:
                conv = Conv1D(
                    num_filters,
                    kernel_size=size,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal',
                    # 临时移除正则化器以测试是否兼容
                    name=f'conv_{size}'
                )(sequence_output)

                conv = BatchNormalization(name=f'bn_conv_{size}')(conv)
                pool = GlobalMaxPooling1D(name=f'pool_{size}')(conv)
                conv_blocks.append(pool)

            # 合并卷积结果
            textcnn_output = Concatenate(name='textcnn_concat')(conv_blocks)
            textcnn_output = Dropout(dropout_rate, name='textcnn_dropout')(textcnn_output)

            # ===== 后续层处理 =====
            emotion_features = Dense(384, activation='relu',
                                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4),
                                     name="emotion_dense")(emotion_input)
            emotion_features = BatchNormalization(name="emotion_norm")(emotion_features)
            emotion_features = Dropout(0.1, name="emotion_dropout")(emotion_features)

            textcnn_proj = Dense(384, activation='relu',
                                 kernel_initializer='glorot_uniform',
                                 name="textcnn_proj")(textcnn_output)
            textcnn_proj = BatchNormalization(name="textcnn_norm")(textcnn_proj)

            emotion_proj = Dense(384, activation='relu',
                                 kernel_initializer='glorot_uniform',
                                 name="emotion_proj")(emotion_features)
            emotion_proj = BatchNormalization(name="emotion_proj_norm")(emotion_proj)

            text_gate = Dense(384, activation='sigmoid',
                              kernel_initializer='glorot_uniform',
                              name="text_gate")(textcnn_proj)
            gated_emotion = Multiply(name="gate_emotion")([emotion_proj, text_gate])

            emotion_gate = Dense(384, activation='sigmoid',
                                 kernel_initializer='glorot_uniform',
                                 name="emotion_gate")(emotion_proj)
            gated_text = Multiply(name="gate_text")([textcnn_proj, emotion_gate])

            combined = Add(name="fusion_add")([textcnn_proj, gated_emotion, gated_text])
            combined = BatchNormalization(name="fusion_combined_norm")(combined)

            combined_reshaped = Reshape((1, 384), name="reshape_for_attention")(combined)

            attn_output = MultiHeadAttention(
                num_heads=4,
                key_dim=24,
                name="multi_head_attention"
            )(combined_reshaped, combined_reshaped)
            attn_output = LayerNormalization(name="attn_norm")(attn_output)

            attn_combined = Add(name="attn_residual")([combined_reshaped, attn_output])
            attn_combined = LayerNormalization(name="attn_combined_norm")(attn_combined)

            combined = Flatten(name="flatten_after_attention")(attn_combined)

            x = Dense(256, activation='relu',
                      kernel_initializer=Orthogonal(),
                      kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4),
                      name="classifier_dense1")(combined)
            x = BatchNormalization(name="classifier_norm1")(x)
            x = Dropout(0.15, name="classifier_dropout1")(x)

            residual = x
            x = Dense(256, activation='relu',
                      kernel_initializer=Orthogonal(),
                      kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4),
                      name="classifier_dense2")(x)
            x = BatchNormalization(name="classifier_norm2")(x)
            x = Dropout(0.15, name="classifier_dropout2")(x)

            x = Add(name="classifier_residual_add")([residual, x])
            x = BatchNormalization(name="classifier_residual_norm")(x)
            x = Activation('relu', name="classifier_residual_activation")(x)

            x = Dense(128, activation='relu',
                      kernel_initializer=Orthogonal(),
                      kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4),
                      name="classifier_dense3")(x)
            x = BatchNormalization(name="classifier_norm3")(x)
            x = Dropout(0.1, name="classifier_dropout3")(x)

            output = Dense(1, activation='sigmoid', dtype='float32', name="output")(x)

        # 构建模型
        model = Model(
            inputs=[input_ids, attention_mask, emotion_input],
            outputs=output,
            name="Police_Enhanced_BERT_TextCNN"
        )

        # 分层微调策略
        bert_model.trainable = True
        for layer in bert_model.layers:
            if layer.name == "bert":
                for i, inner_layer in enumerate(layer.encoder.layer):
                    inner_layer.trainable = (i >= 4)
                layer.pooler.trainable = False

        print("\n模型层信息:")
        for i, layer in enumerate(model.layers):
            trainable_status = "可训练" if layer.trainable else "冻结"
            print(f"{i:2d} {layer.name:25s} {trainable_status}")

        return model, 'bert-textcnn-police-enhanced'

    except Exception as e:
        print(f"模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 使用TensorFlow Datasets加速数据加载
def create_tf_dataset(inputs, labels, batch_size=24, buffer_size=1000, shuffle=True):
    """创建高性能TensorFlow数据集"""
    if isinstance(inputs, dict) and 'emotion_input' in inputs:
        inputs['emotion_input'] = tf.cast(inputs['emotion_input'], tf.float32)
    if isinstance(inputs, dict):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((tuple(inputs) if isinstance(inputs, list) else inputs, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# t-SNE可视化函数
def plot_tsne(embeddings, labels, title="t-SNE Visualization", save_path="tsne_visualization.png"):
    """
    使用t-SNE算法将高维数据降维并可视化

    参数:
    embeddings - 高维特征向量 (n_samples, n_features)
    labels - 每个样本的标签 (n_samples,)
    title - 图表标题
    save_path - 保存图片的路径
    """
    print("开始t-SNE降维...")

    # 随机采样1000个点用于可视化（提高速度）
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[indices]
    sample_labels = labels[indices]

    # 使用t-SNE降维到二维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(sample_embeddings)

    print("绘制t-SNE可视化图表...")
    plt.figure(figsize=(10, 8))

    # 为不同类别的样本设置不同的颜色
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=sample_labels, cmap='viridis', alpha=0.6,
                          edgecolors='w', s=40)

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class Label', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存并显示图表
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE可视化图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    jieba_cache = {}  # 重置分词缓存

    # 加载数据
    train_data, test_data, stopwords = load_and_preprocess_data()
    if train_data is None:
        exit(1)

    print(f"数据加载和预处理耗时: {time.time() - start_time:.2f}秒")

    # 构建词汇表
    print("构建词汇表...")
    all_docs = train_data['processed'].tolist() + test_data['processed'].tolist()
    vocab = set(word for doc in all_docs for word in doc)
    print(f"词汇表大小: {len(vocab)}")

    # 加载情感词典 - 增强版（包含公安信访词典）
    positive_words, negative_words, emotion_intensity = load_emotion_dicts(vocab)
    print(f"总积极情感词数量: {len(positive_words)}, 总消极情感词数量: {len(negative_words)}")

    # 加载BERT tokenizer - 强制使用BERT
    bert_path = "/root/autodl-fs/part2/gai/bert-base-chinese"
    try:
        if os.path.exists(bert_path):
            tokenizer = BertTokenizer.from_pretrained(bert_path)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        print("BERT tokenizer加载成功")
    except Exception as e:
        print(f"加载BERT tokenizer失败: {e}")
        print("错误: 必须使用BERT模型，程序退出")
        sys.exit(1)

    max_len = 128  # 固定序列长度


    # 文本编码函数
    def encode_texts(texts):
        """优化文本编码函数"""
        return tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True
        )


    # 处理文本数据
    print("处理文本数据...")
    train_texts = [' '.join(doc) for doc in train_data['processed']]
    test_texts = [' '.join(doc) for doc in test_data['processed']]

    # 使用批处理加速文本编码
    print("批量编码文本...")
    train_encodings = encode_texts(train_texts)
    test_encodings = encode_texts(test_texts)

    X_train_ids = train_encodings['input_ids']
    X_train_mask = train_encodings['attention_mask']
    X_test_ids = test_encodings['input_ids']
    X_test_mask = test_encodings['attention_mask']

    # 增强情感特征提取
    print("并行提取增强情感特征...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        X_train_emo = list(executor.map(
            lambda doc: extract_emotional_features(doc, positive_words, negative_words, emotion_intensity),
            train_data['processed']
        ))
        X_test_emo = list(executor.map(
            lambda doc: extract_emotional_features(doc, positive_words, negative_words, emotion_intensity),
            test_data['processed']
        ))

    X_train_emo = np.array(X_train_emo, dtype=np.float32)
    X_test_emo = np.array(X_test_emo, dtype=np.float32)
    print(f"情感特征维度: {X_train_emo.shape[1]}")

    # 特征标准化
    emo_scaler = StandardScaler()
    X_train_emo_scaled = emo_scaler.fit_transform(X_train_emo)
    X_test_emo_scaled = emo_scaler.transform(X_test_emo)

    # 获取标签
    y_train = train_data['label'].values.astype(np.float32)
    y_test = test_data['label'].values.astype(np.float32)

    # 添加调试输出
    print("\n" + "=" * 50)
    print(f"正样本比例: {np.mean(y_train):.6f}")
    print(f"负样本比例: {1 - np.mean(y_train):.6f}")
    print("情感特征示例:", X_train_emo_scaled[0])

    # 检查输入内容
    sample_text = tokenizer.decode(X_train_ids[0].numpy())
    sample_words = train_data.iloc[0]['processed']
    print(f"样本文本: {sample_text}")
    print(f"分词结果: {sample_words}")
    print(f"包含标点: {'!' in sample_words or '？' in sample_words}")
    print(f"对应标签: {y_train[0]}")
    print("=" * 50 + "\n")

    # 构建高性能模型
    print("构建公安信访情感分析BERT+TextCNN模型...")
    model, model_type = build_enhanced_bert_textcnn_model(
        max_len,
        emo_dim=X_train_emo_scaled.shape[1],
        dropout_rate=0.2
    )

    if model is None:
        print("错误: 模型构建失败，程序退出")
        sys.exit(1)

    # 优化训练配置
    batch_size = 24
    base_lr = 2e-5  # 降低学习率

    # 使用标准Adam优化器
    optimizer = Adam(
        learning_rate=base_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # 编译模型 - 添加更多评估指标
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(from_logits=False),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 F1Score(name='f1_score')]
    )

    print(f"模型类型: {model_type}")
    print(model.summary())

    # 回调函数 - 优化训练策略
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # 添加TensorBoard回调
    tensorboard = TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        update_freq='epoch'
    )

    # 正确划分验证集
    train_indices, val_indices = train_test_split(
        np.arange(len(train_data)),
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )

    # 准备训练和验证数据
    X_val_ids = tf.gather(X_train_ids, val_indices)
    X_val_mask = tf.gather(X_train_mask, val_indices)
    X_val_emo_scaled = tf.convert_to_tensor(X_train_emo_scaled[val_indices], dtype=tf.float32)
    y_val = y_train[val_indices]

    X_train_ids = tf.gather(X_train_ids, train_indices)
    X_train_mask = tf.gather(X_train_mask, train_indices)
    X_train_emo_scaled = tf.convert_to_tensor(X_train_emo_scaled[train_indices], dtype=tf.float32)
    y_train = y_train[train_indices]

    # 准备训练和验证数据
    train_inputs = {
        'input_ids': X_train_ids,
        'attention_mask': X_train_mask,
        'emotion_input': X_train_emo_scaled
    }
    val_inputs = {
        'input_ids': X_val_ids,
        'attention_mask': X_val_mask,
        'emotion_input': X_val_emo_scaled
    }

    # 创建TensorFlow数据集
    train_dataset = create_tf_dataset(train_inputs, y_train, batch_size=batch_size)
    val_dataset = create_tf_dataset(val_inputs, y_val, batch_size=batch_size, shuffle=False)

    # 训练模型
    print("开始训练公安信访情感分析模型...")
    print("提示: 按Ctrl+C可手动停止训练并评估当前最佳模型")
    try:
        history = model.fit(
            train_dataset,
            epochs=5,  # 训练轮次
            validation_data=val_dataset,
            callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard],
            verbose=1
        )
    except KeyboardInterrupt:
        print("\n训练被手动中断! 正在加载最佳模型并评估...")

    # 评估模型
    print("评估模型...")
    if os.path.exists('best_model.h5'):
        model.load_weights('best_model.h5')
        print("已加载最佳模型权重")
    else:
        print("警告: 未找到最佳模型文件，使用当前模型进行评估")

    # 准备测试数据
    test_inputs = {
        'input_ids': X_test_ids,
        'attention_mask': X_test_mask,
        'emotion_input': X_test_emo_scaled
    }

    # 使用批量预测
    y_pred_prob = model.predict(test_inputs,  batch_size=batch_size * 2)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("公安信访情感分析模型评测结果:")
    print(f"模型类型: {model_type.upper()}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精准率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1值 (F1-Score): {f1:.4f}")
    print("=" * 50)

    # t-SNE可视化
    print("\n准备进行t-SNE可视化...")

    # 创建特征提取模型 - 提取融合层的输出
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer('classifier_norm3').output  # 分类层前的特征
    )

    # 随机选择1000个测试样本进行可视化
    n_samples = min(1000, len(X_test_ids))
    sample_indices = np.random.choice(len(X_test_ids), n_samples, replace=False)

    # 准备特征提取输入
    sample_inputs = {
        'input_ids': tf.gather(X_test_ids, sample_indices),
        'attention_mask': tf.gather(X_test_mask, sample_indices),
        'emotion_input': tf.gather(X_test_emo_scaled, sample_indices)
    }

    # 提取高维特征
    print("提取高维特征向量...")
    embeddings = feature_extractor.predict(sample_inputs, batch_size=batch_size)

    # 提取对应标签
    sample_labels = y_test[sample_indices]

    # 执行t-SNE可视化
    plot_tsne(
        embeddings,
        sample_labels,
        title=f"BERT+TextCNN+情感特征融合表示 (n={n_samples})",
        save_path="bert_textcnn_emotion_tsne.png"
    )

    # 保存模型
    model_save_name = f'police_emotion_bert_textcnn_model.keras'
    print(f"\n保存高性能模型: {model_save_name}")
    model.save(model_save_name)

    # 保存性能报告
    report = f"""===== 公安信访情感分析模型报告 =====
模型类型: {model_type.upper()}
准确率: {accuracy:.4f}
精准率: {precision:.4f}
召回率: {recall:.4f}
F1值: {f1:.4f}
训练时间: {time.time() - start_time:.2f}秒
使用技术: 
- BERT + TextCNN + 情感增强特征
- 公安信访专用情感词典
- 双向门控融合机制
- 多头自注意力机制
优化措施:
- 情感特征维度: {X_train_emo_scaled.shape[1]}
- 批处理大小: {batch_size}
- 学习率: {base_lr}
- 训练轮次: 5
"""
    with open('police_emotion_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n总耗时: {time.time() - start_time:.2f}秒")
    print("公安信访情感分析模型训练完成")
