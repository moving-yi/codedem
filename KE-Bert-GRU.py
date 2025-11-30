import re
import os
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalMaxPooling1D, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import jieba
from transformers import BertTokenizer, TFBertModel
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# 添加t-SNE可视化所需的库
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 完全禁用GPU
tf.config.set_visible_devices([], 'GPU')

# 配置日志系统
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

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
    """高效文本预处理函数"""
    if not isinstance(text, str) or pd.isna(text):
        return []

    if text in jieba_cache:
        return jieba_cache[text]

    clean_text = re.sub(r'\d+', 'NUM', text)
    words = [word for word in jieba.lcut(clean_text) if word.strip()]
    jieba_cache[text] = words
    return words


# 1. 数据加载与预处理
def load_and_preprocess_data():
    """高效加载和预处理数据"""
    print("加载数据...")
    try:
        # 尝试不同的编码
        for encoding in ['gbk', 'utf-8', 'latin1']:
            try:
                data = pd.read_csv('data.csv', header=None,
                                   encoding=encoding,
                                   names=['text', 'label'],
                                   usecols=[0, 1],
                                   engine='python',
                                   on_bad_lines='skip')
                print(f"使用{encoding}编码成功加载数据")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法确定文件编码")

        data = data.dropna(subset=['text', 'label']).copy()
        data['text'] = data['text'].str.strip()

    except FileNotFoundError:
        print("错误: 找不到data.csv文件")
        return None, None, None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

    # 加载停用词
    stopwords_set = set()
    try:
        if os.path.exists('stopwords.txt'):
            with open('stopwords.txt', encoding='utf-8') as f:
                stopwords_set = set(line.strip() for line in f if line.strip())
                print(f"加载{len(stopwords_set)}个停用词")
        else:
            print("警告: 未找到stopwords.txt文件，将使用空停用词表")
    except Exception as e:
        print(f"停用词加载失败: {e}")

    # 划分数据集
    try:
        train_data, test_data = train_test_split(
            data,
            test_size=0.15,
            stratify=data['label'],
            random_state=42
        )
    except ValueError as e:
        print(f"数据集划分失败: {e}")
        train_data = data.sample(frac=0.85, random_state=42)
        test_data = data.drop(train_data.index)
        print("使用备选方式划分数据集")

    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

    # 使用并行处理加速文本预处理
    print("并行预处理文本...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        train_data['processed'] = list(executor.map(
            lambda x: efficient_preprocess(x, stopwords_set),
            train_data['text']
        ))
        test_data['processed'] = list(executor.map(
            lambda x: efficient_preprocess(x, stopwords_set),
            test_data['text']
        ))

    # 过滤空文本
    train_data = train_data[train_data['processed'].str.len() > 0]
    test_data = test_data[test_data['processed'].str.len() > 0]

    # 添加：过滤短文本
    train_data = train_data[train_data['processed'].apply(len) >= 3]
    test_data = test_data[test_data['processed'].apply(len) >= 3]

    return train_data, test_data, stopwords_set


# 加载情感词典（包括普通词典和公安信访词典）
def load_emotion_dicts(vocab):
    """加载情感词典并优化存储"""
    if not hasattr(load_emotion_dicts, 'cache'):
        load_emotion_dicts.cache = {}

    cache_key = tuple(sorted(vocab))
    if cache_key in load_emotion_dicts.cache:
        return load_emotion_dicts.cache[cache_key]

    positive_words = set()
    negative_words = set()
    emotion_intensity = {}

    # 1. 加载普通情感词典
    try:
        with open('NTUSD+my_pos.txt', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    emotion_intensity[word] = 2.0  # 普通情感词强度
                    positive_words.add(word)
        print(f"加载普通积极词典: {len(positive_words)}个词")
    except FileNotFoundError:
        print("警告: 未找到普通积极情感词典文件")

    try:
        with open('NTUSD+my_neg.txt', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    emotion_intensity[word] = 2.0  # 普通情感词强度
                    negative_words.add(word)
        print(f"加载普通消极词典: {len(negative_words)}个词")
    except FileNotFoundError:
        print("警告: 未找到普通消极情感词典文件")

    # 2. 加载公安信访情感词典
    try:
        with open('police_pos.txt', encoding='utf-8') as f:
            police_pos = set()
            for line in f:
                word = line.strip()
                if word:
                    emotion_intensity[word] = 4.0  # 公安信访情感词强度更高
                    police_pos.add(word)
            positive_words |= police_pos  # 合并到积极词集
            print(f"加载公安信访积极词典: {len(police_pos)}个词")
    except FileNotFoundError:
        print("警告: 未找到公安信访积极词典文件")

    try:
        with open('police_neg.txt', encoding='utf-8') as f:
            police_neg = set()
            for line in f:
                word = line.strip()
                if word:
                    emotion_intensity[word] = 4.0  # 公安信访情感词强度更高
                    police_neg.add(word)
            negative_words |= police_neg  # 合并到消极词集
            print(f"加载公安信访消极词典: {len(police_neg)}个词")
    except FileNotFoundError:
        print("警告: 未找到公安信访消极词典文件")

    # 保留在词汇表中的情感词
    positive_words = positive_words & vocab
    negative_words = negative_words & vocab

    print(f"总积极情感词数量: {len(positive_words)}, 总消极情感词数量: {len(negative_words)}")

    result = (positive_words, negative_words, emotion_intensity)
    load_emotion_dicts.cache[cache_key] = result
    return result


# 增强情感特征提取
def extract_emotional_features(words, positive_words, negative_words, emotion_intensity):
    """提取关键情感特征"""
    if not words:
        return np.zeros(14, dtype=np.float32)

    word_count = len(words)
    pos_count = 0
    neg_count = 0
    emotion_scores = []
    emotion_positions = []

    for i, word in enumerate(words):
        if word in positive_words:
            pos_count += 1
            intensity = emotion_intensity.get(word, 1.0)
            emotion_scores.append(intensity)
            position_weight = 1.0 + 0.5 * (1 - abs(2 * i / word_count - 1))
            emotion_positions.append(position_weight)
        elif word in negative_words:
            neg_count += 1
            intensity = -emotion_intensity.get(word, 1.0)
            emotion_scores.append(intensity)
            position_weight = 1.0 + 0.5 * (1 - abs(2 * i / word_count - 1))
            emotion_positions.append(position_weight)

    sentiment_sum = sum(emotion_scores) if emotion_scores else 0.0
    sentiment_avg = sentiment_sum / max(len(emotion_scores), 1) if emotion_scores else 0.0
    emotion_variance = np.var(emotion_scores) if len(emotion_scores) > 1 else 0.0

    if len(emotion_scores) > 1:
        diff = [emotion_scores[i + 1] - emotion_scores[i] for i in range(len(emotion_scores) - 1)]
        intensity_change = sum(diff) / len(diff)
    else:
        intensity_change = 0.0

    exclamation = words.count('!') + words.count('！')
    question = words.count('?') + words.count('？')

    negation_words = {'不', '没', '无', '非', '未', '莫', '勿', '毋'}
    negation_count = sum(1 for word in words if word in negation_words)

    weighted_sentiment = sum(s * p for s, p in zip(emotion_scores, emotion_positions)) / max(sum(emotion_positions), 1)

    pos_positions = [i / len(words) for i, word in enumerate(words) if word in positive_words]
    neg_positions = [i / len(words) for i, word in enumerate(words) if word in negative_words]

    pos_position_mean = np.mean(pos_positions) if pos_positions else 0.0
    neg_position_mean = np.mean(neg_positions) if neg_positions else 0.0
    pos_position_var = np.var(pos_positions) if len(pos_positions) > 1 else 0.0
    emotion_density = len(emotion_scores) / max(len(words), 1)

    return np.array([
        pos_count, neg_count, sentiment_avg, weighted_sentiment,
        emotion_variance, intensity_change, sentiment_sum / max(word_count, 1),
        exclamation, question, negation_count, pos_position_mean,
        neg_position_mean, pos_position_var, emotion_density
    ], dtype=np.float32)


# 简化模型结构 - 修改为返回特征提取模型
def build_bert_gru_model(max_len, emo_dim, dropout_rate=0.2):
    """构建CPU友好的BERT+GRU模型"""
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    emotion_input = Input(shape=(emo_dim,), name='emotion_input', dtype=tf.float32)

    try:
        print("加载优化的BERT模型...")
        model_path = "./bert-base-chinese"

        if os.path.exists(model_path):
            bert_model = TFBertModel.from_pretrained(model_path, add_pooling_layer=False)
        else:
            bert_model = TFBertModel.from_pretrained("bert-base-chinese", add_pooling_layer=False)

        # 关键修复：在构建模型前冻结BERT底层
        bert_model.trainable = True
        for layer in bert_model.layers:
            if layer.name == "bert":
                for i, inner_layer in enumerate(layer.encoder.layer):
                    # 只训练最后3层，冻结前9层
                    inner_layer.trainable = (i >= 9) if len(layer.encoder.layer) > 11 else (i >= 8)

        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        print("BERT模型加载成功")

        # 简化GRU层
        gru_layer = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name="gru_layer")(
            sequence_output)
        # 使用GlobalMaxPooling1D提取特征
        text_features = GlobalMaxPooling1D(name="text_features")(gru_layer)

        # 简化情感特征处理
        emotion_features = Dense(128, activation='relu', name='emotion_features')(emotion_input)
        emotion_features = Dropout(0.1, name='emotion_dropout')(emotion_features)

        # 特征融合
        combined = Concatenate(name='feature_fusion')([text_features, emotion_features])
        combined = Dense(256, activation='relu', name='fusion_dense')(combined)
        combined = Dropout(dropout_rate, name='fusion_dropout')(combined)

        # 分类层
        x = Dense(128, activation='relu', name='classifier_dense')(combined)
        x = Dropout(dropout_rate, name='classifier_dropout')(x)
        output = Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

        # 构建用于训练的模型
        train_model = Model(
            inputs=[input_ids, attention_mask, emotion_input],
            outputs=output,
            name="BERT_GRU_CPU"
        )

        # 构建用于特征提取的模型（输出为GRU层的特征表示）
        feature_extractor = Model(
            inputs=[input_ids, attention_mask, emotion_input],
            outputs=text_features  # 只提取文本特征
        )

        return train_model, feature_extractor

    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 使用TensorFlow Datasets加速数据加载
def create_tf_dataset(inputs, labels, batch_size=32, buffer_size=1000, shuffle=True):
    if isinstance(inputs, dict):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((tuple(inputs) if isinstance(inputs, list) else inputs, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


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
    cbar.set_label('情感标签 (0=消极, 1=积极)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存并显示图表
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE可视化图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("当前使用设备：CPU")
    start_time = time.time()
    jieba_cache = {}

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

    # 加载情感词典（普通词典和公安信访词典）
    print("加载情感词典...")
    positive_words, negative_words, emotion_intensity = load_emotion_dicts(vocab)

    # 加载BERT tokenizer
    bert_path = "./bert-base-chinese"
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_path if os.path.exists(bert_path) else "bert-base-chinese")
        print("BERT tokenizer加载成功")
    except Exception as e:
        print(f"加载BERT tokenizer失败: {e}")
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

    # 分批处理防止OOM
    batch_size_text = 256
    train_encodings = []
    for i in range(0, len(train_texts), batch_size_text):
        batch = train_texts[i:i + batch_size_text]
        train_encodings.append(encode_texts(batch))

    # 合并批处理结果
    train_encodings_merged = {
        'input_ids': tf.concat([e['input_ids'] for e in train_encodings], axis=0),
        'attention_mask': tf.concat([e['attention_mask'] for e in train_encodings], axis=0)
    }

    test_encodings = encode_texts(test_texts)

    X_train_ids = train_encodings_merged['input_ids']
    X_train_mask = train_encodings_merged['attention_mask']
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

    # 构建模型
    print("构建BERT+GRU公安信访情感分析模型(CPU版)...")
    train_model, feature_extractor = build_bert_gru_model(
        max_len,
        emo_dim=X_train_emo_scaled.shape[1],
        dropout_rate=0.2
    )

    if train_model is None:
        sys.exit(1)

    # 优化训练配置
    batch_size = 16
    base_lr = 2e-5

    # 使用Adam优化器
    optimizer = Adam(learning_rate=base_lr)

    # 编译模型
    train_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', F1Score(name='f1_score')]
    )

    print(train_model.summary())

    # 回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'best_model_cpu.keras',
        monitor='val_f1_score',
        save_best_only=True,
        save_weights_only=True,  # 只保存权重，避免模型结构问题
        mode='max',
        verbose=1
    )

    # 划分验证集
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
    train_dataset = create_tf_dataset(train_inputs, y_train, batch_size=batch_size, buffer_size=500)
    val_dataset = create_tf_dataset(val_inputs, y_val, batch_size=batch_size, shuffle=False)

    # 训练模型
    print("开始训练(CPU模式)...")
    history = train_model.fit(
        train_dataset,
        epochs=5,
        validation_data=val_dataset,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # 评估模型 - 修复权重加载问题
    print("评估模型...")
    if os.path.exists('best_model_cpu.keras'):
        try:
            # 尝试加载权重而不是整个模型
            print("加载最佳模型权重...")
            train_model.load_weights('best_model_cpu.keras')
            print("权重加载成功")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print("使用训练结束时的模型进行评估")
    else:
        print("使用训练结束时的模型进行评估")

    # 准备测试数据
    test_inputs = {
        'input_ids': X_test_ids,
        'attention_mask': X_test_mask,
        'emotion_input': X_test_emo_scaled
    }

    # 预测
    y_pred_prob = train_model.predict(test_inputs, batch_size=64)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("模型评测结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精准率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print("=" * 50)

    # ===== 添加t-SNE可视化分析 =====
    print("\n开始进行t-SNE可视化分析...")

    # 1. 提取测试集文本特征
    print("提取测试集文本特征...")
    text_embeddings = feature_extractor.predict(test_inputs, batch_size=batch_size)

    # 2. 随机采样部分数据用于可视化（避免计算量过大）
    n_samples = min(1000, len(text_embeddings))
    sample_indices = np.random.choice(len(text_embeddings), n_samples, replace=False)

    # 3. 执行t-SNE降维并可视化
    plot_tsne(
        embeddings=text_embeddings[sample_indices],
        labels=y_test[sample_indices],
        title="BERT+GRU公安信访情感分析特征空间",
        save_path="bert_gru_tsne.png"
    )

    # 保存完整模型
    model_save_name = f'police_emotion_model_cpu.keras'
    train_model.save(model_save_name)

    # 添加模型信息到报告
    model_report = f"""
===== 公安信访情感分析模型报告 =====
模型架构: BERT+GRU
准确率: {accuracy:.4f}
精准率: {precision:.4f}
召回率: {recall:.4f}
F1值: {f1:.4f}
训练时间: {time.time() - start_time:.2f}秒
特征可视化: bert_gru_tsne.png
"""
    with open('model_report.txt', 'w') as f:
        f.write(model_report)

    print(f"\n总耗时: {time.time() - start_time:.2f}秒")
    print("训练完成")
