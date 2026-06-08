# 基于IT行业招聘数据的个性化岗位推荐系统模型


```python
#导入库和数据
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
df=pd.read_excel(r"D:\ASUS\Desktop\2025-2026two\machine learning\it行业招聘数据.xlsx")
```


```python
#数据清洗
df_clean = df[['岗位id', '岗位名', '岗位描述', '岗位标签', '检索二级职位类别', '检索三级职位类别',
               '检索城市', '工作年限要求', '学历要求', '公司规模', '公司类型', '公司行业',
               '最大薪资', '最小薪资', '是否全职', '公司名称', '岗位发布-lat', '岗位发布-lon']].copy()
df_clean = df_clean.dropna(subset=['岗位描述', '检索二级职位类别', '最大薪资', '最小薪资'])
df_clean['平均薪资'] = (df_clean['最大薪资'] + df_clean['最小薪资']) / 2
# 过滤异常薪资（去除过高和过低的）
df_clean = df_clean[(df_clean['平均薪资'] >= 2000) & (df_clean['平均薪资'] <= 300000)]
```


```python
print(f"清洗后数据量: {len(df_clean):,}")
print(f"\n职位类别分布:")
print(df_clean['检索二级职位类别'].value_counts())
print(f"\n城市分布(Top 10):")
print(df_clean['检索城市'].value_counts().head(10))
```

    清洗后数据量: 102,717
    
    职位类别分布:
    检索二级职位类别
    运维/技术支持    23185
    后端开发       21777
    测试         17048
    技术管理       15506
    销售技术支持      8703
    人工智能        8151
    前端/移动开发     4461
    数据          3886
    Name: count, dtype: int64
    
    城市分布(Top 10):
    检索城市
    上海    17243
    深圳    15496
    广州    10470
    北京     5956
    杭州     4990
    苏州     4973
    武汉     3620
    成都     3267
    东莞     3230
    南京     2221
    Name: count, dtype: int64
    


```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import ahocorasick
import re

# ==========================================
# 1. 数据准备与预处理
# ==========================================

# 【修复 1】重置索引，防止稀疏矩阵构建时报错
df_clean = df_clean.reset_index(drop=True)

# 填充空值，统一转为小写，去除首尾空格
# 这一步对于解决大小写不匹配至关重要
df_clean['岗位描述'] = df_clean['岗位描述'].fillna('').astype(str)
df_clean['clean_text'] = df_clean['岗位描述'].str.lower().str.strip()

# ==========================================
# 2. 构建增强型技能关键词库
# ==========================================

# 【关键优化】扩充后的泛技术技能词库 (建议保存为外部 CSV/JSON 以便维护)
# 覆盖了研发、运维、测试、数据、产品、设计等岗位
SKILL_KEYWORDS_RAW = [
    # --- 编程语言 ---
    'java', 'python', 'c++', 'c#', 'javascript', 'typescript', 'go', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r语言', 'lua', 'perl', 'shell', 'matlab', 'sas', 'spss',
    # --- 前端/移动端 ---
    'html', 'css', 'vue', 'react', 'angular', 'jquery', 'node.js', 'webpack', 'bootstrap', '小程序', 'uni-app', 'flutter', 'react native', 'ios', 'android',
    # --- 后端/架构 ---
    'spring boot', 'spring cloud', 'mybatis', 'hibernate', 'django', 'flask', 'fastapi', 'gin', 'beego', '.net', 'asp.net', 'mvc', '微服务', '分布式', '高并发',
    # --- 数据库/中间件 ---
    'mysql', 'oracle', 'sql server', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'zookeeper', 'nginx', 'tomcat', 'apache', 'iis',
    # --- 运维/DevOps ---
    'linux', 'docker', 'kubernetes', 'k8s', 'jenkins', 'git', 'svn', 'ansible', 'saltstack', 'puppet', 'aws', 'azure', '阿里云', '腾讯云', '华为云', 'shell脚本',
    # --- 大数据/AI ---
    'hadoop', 'spark', 'flink', 'hive', 'hbase', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'opencv', 'nlp', '计算机视觉', '深度学习', '机器学习', '数据挖掘',
    # --- 测试/质量 ---
    'selenium', 'jmeter', 'loadrunner', 'postman', 'appium', '自动化测试', '性能测试', '接口测试', '黑盒测试', '白盒测试',
    # --- 通用/办公/软技能 ---
    'office', 'excel', 'word', 'ppt', 'visio', 'cad', 'photoshop', 'axure', 'xmind', '项目管理', '需求分析', '英语', '日语', '沟通', '团队协作'
    # === 工业体系与标准 ===
    'iso9001', 'iso14001', 'iso45001', 'haccp', 'gmp', 'fmea', '5s', 'tpm', '六西格玛',
    '质量管理体系', '环境管理体系', '职业健康安全', '合规性评价', '文件控制',

     # === 设备运维与自动化 ===
     'plc', 'scada', 'dcs', 'rtu', '工控机', '变频器', '伺服电机', '传感器', '继电器',
     '点检', '巡检', '预防性维护', '故障诊断', '备件管理', '维修工单',
     '高低压证', '电工证', '特种作业操作证', '登高证', '焊工证',

     # === 新能源/电力相关 ===
     '风电', '光伏', '储能', '逆变器', '升压站', '配电柜', '变压器', '断路器',
     '继电保护', '电网调度', 'agc', 'avc', 'ems', 'scada系统',

     # === 测试/质检专用工具 ===
     'canoe', 'wireshark', 'vector', 'capl', 'udsoncan', 'j1939', 'doip',
     'labview', 'teststand', 'ni', 'keysight', '福禄克', '德图',

     # === 通用能力词（兜底模糊描述）===
     '系统集成', '数字化转型', '信息化建设', '报表开发', '数据治理', '流程优化'
]

# 建立别名映射字典 (解决 "C++" vs "c++" vs "cplusplus" 等问题)
# 键：原始标准名 (用于矩阵列名)
# 值：需要匹配的变体列表 (全部小写)
ALIAS_MAP = {
    'Java': ['java'],
    'Python': ['python', 'py'],
    'C++': ['c++', 'cpp', 'cplusplus'],
    'C#': ['c#', 'csharp', 'c sharp'],
    'JavaScript': ['javascript', 'js', 'ecmascript'],
    'Node.js': ['node.js', 'nodejs', 'node'],
    'TypeScript': ['typescript', 'ts'],
    'SQL': ['sql', 'structured query language'],  # 注意：不再包含 mysql/oracle，避免重复计数
    'MySQL': ['mysql', 'mariadb'],
    'Oracle': ['oracle', 'oracle db'],
    'PostgreSQL': ['postgresql', 'postgres'],
    'Redis': ['redis'],
    'MongoDB': ['mongodb', 'mongo'],
    'Linux': ['linux', 'ubuntu', 'centos', 'redhat', 'debian'],
    'Docker': ['docker', 'container'],
    'Kubernetes': ['kubernetes', 'k8s'],
    'Git': ['git', 'github', 'gitlab'],
    'Office': ['office', 'excel', 'word', 'powerpoint', 'ppt', 'wps'],
    'CAD': ['cad', 'autocad', 'solidworks'],
    'Photoshop': ['photoshop', 'ps', 'adobe ps'],
    'PLC': ['plc', '可编程逻辑控制器', '梯形图', 's7-200', 's7-300'],
    'ISO9001': ['iso9001', '质量体系', '体系文件', '内审', '外审'],
    'CANoe': ['canoe', 'vector canoe', '总线仿真']
}
# 展平所有需要匹配的关键词，并记录它们对应的“标准列名”
all_patterns = []
pattern_to_col = {}

for std_name, aliases in ALIAS_MAP.items():
    for alias in aliases:
        clean_alias = alias.strip().lower()
        if clean_alias and clean_alias not in pattern_to_col:
            all_patterns.append(clean_alias)
            pattern_to_col[clean_alias] = std_name

# 如果还有没在 ALIAS_MAP 里的词，直接加入
for word in SKILL_KEYWORDS_RAW:
    w = word.strip().lower()
    if w and w not in pattern_to_col:
        all_patterns.append(w)
        pattern_to_col[w] = word.capitalize() # 简单首字母大写作为列名

# 去重并排序 (可选，为了稳定性)
unique_patterns = list(set(all_patterns))

print(f"🚀 正在构建自动机，共加载 {len(unique_patterns)} 个匹配模式...")

# ==========================================
# 3. 构建 AC 自动机
# ==========================================

automaton = ahocorasick.Automaton()

for idx, keyword in enumerate(unique_patterns):
    # 使用 keyword 本身作为 key，value 为 (idx, keyword)
    automaton.add_word(keyword, (idx, keyword))

automaton.make_automaton()
print("✅ 自动机构建完成")

# ==========================================
# 4. 扫描数据并生成稀疏矩阵 (含最长匹配逻辑)
# ==========================================

num_docs = len(df_clean)
num_skills = len(unique_patterns) # 这里的维度是“匹配模式”的数量

rows = []
cols = []

# 遍历每一行数据
for i, text in enumerate(df_clean['clean_text']):
    if not text:
        continue

    # 查找所有匹配项
    # end_index 是结束位置，(col_idx, matched_keyword) 是存储的值
    found_indices = set() # 用 set 防止同一行同一个技能被多次计数

    # 获取当前文本的所有匹配结果
    matches = list(automaton.iter(text))

    # 【关键修复】处理 Java / JavaScript 冲突
    # 策略：如果一个短词的匹配范围完全被一个长词包含，则丢弃短词
    valid_matches = []
    for end_idx_short, (col_idx_short, kw_short) in matches:
        is_substring = False
        start_idx_short = end_idx_short - len(kw_short) + 1

        for end_idx_long, (col_idx_long, kw_long) in matches:
            if col_idx_short == col_idx_long:
                continue # 跳过自己

            start_idx_long = end_idx_long - len(kw_long) + 1

            # 检查 short 是否被 long 包含
            if start_idx_long <= start_idx_short and end_idx_short <= end_idx_long:
                is_substring = True
                break

        if not is_substring:
            valid_matches.append((end_idx_short, (col_idx_short, kw_short)))

    # 将清洗后的有效匹配加入列表
    for _, (col_idx, _) in valid_matches:
        found_indices.add(col_idx)

    if found_indices:
        rows.extend([i] * len(found_indices))
        cols.extend(list(found_indices))

print(f"🔍 扫描完成。命中记录数: {len(rows)}")

# ==========================================
# 5. 生成最终矩阵
# ==========================================

# 构建稀疏矩阵
skill_matrix = csr_matrix(
    (np.ones(len(rows), dtype=np.int8), (rows, cols)),
    shape=(num_docs, num_skills)
)

# 打印统计信息
print(f"\n================ 结果统计 ================")
print(f"✅ 样本总数: {num_docs:,}")
print(f"✅ 特征维度: {num_skills} (匹配模式数量)")
print(f"✅ 稀疏矩阵形状: {skill_matrix.shape}")
print(f"✅ 非零元素个数: {skill_matrix.nnz:,}")
print(f"✅ 覆盖率: {skill_matrix.nnz / (num_docs * num_skills) * 100:.4f}%")
print(f"✅ 平均每行命中技能数: {skill_matrix.nnz / num_docs:.2f}")

# 查看未命中的数据量
zero_rows = num_docs - np.count_nonzero(skill_matrix.getnnz(axis=1))
print(f"⚠️ 未命中任何技能的行数: {zero_rows:,} (占比 {zero_rows/num_docs*100:.2f}%)")
```

    🚀 正在构建自动机，共加载 225 个匹配模式...
    ✅ 自动机构建完成
    🔍 扫描完成。命中记录数: 409702
    
    ================ 结果统计 ================
    ✅ 样本总数: 102,717
    ✅ 特征维度: 225 (匹配模式数量)
    ✅ 稀疏矩阵形状: (102717, 225)
    ✅ 非零元素个数: 409,702
    ✅ 覆盖率: 1.7727%
    ✅ 平均每行命中技能数: 3.99
    ⚠️ 未命中任何技能的行数: 8,454 (占比 8.23%)
    


```python
# 4. 结构化特征编码
print("\n📊 结构化特征编码...")
top_cities = df_clean['检索城市'].value_counts().head(15).index.tolist()
df_clean['城市编码'] = df_clean['检索城市'].apply(lambda x: top_cities.index(x) if x in top_cities else len(top_cities))

cat_map = {cat: i for i, cat in enumerate(df_clean['检索二级职位类别'].unique())}
df_clean['职位类别编码'] = df_clean['检索二级职位类别'].map(cat_map)

edu_order = {'初中及以下': 0, '高中': 1, '中技/中专': 2, '大专': 3, '本科': 4, '硕士': 5, '博士': 6}
df_clean['学历编码'] = df_clean['学历要求'].map(edu_order).fillna(4)

def parse_experience(exp_text):
    if pd.isna(exp_text):
        return 3
    exp_text = str(exp_text)
    if '无需经验' in exp_text or '在校生' in exp_text:
        return 0
    if '1年' in exp_text:
        return 1
    if '2年' in exp_text:
        return 2
    if '3年' in exp_text:
        return 3
    if '5年' in exp_text:
        return 5
    nums = re.findall(r'\d+', exp_text)
    if nums:
        return int(nums[0])
    return 3
df_clean['经验编码'] = df_clean['工作年限要求'].apply(parse_experience)
df_clean['薪资分位数'] = df_clean['平均薪资'].rank(pct=True)

struct_features = np.column_stack([
    df_clean['城市编码'].values / len(top_cities),
    df_clean['职位类别编码'].values / len(cat_map),
    df_clean['学历编码'].values / 6,
    df_clean['经验编码'].values / 10,
    df_clean['薪资分位数'].values
])
print(f"✅ 结构化特征完成，维度: {struct_features.shape}")
```

    
    📊 结构化特征编码...
    ✅ 结构化特征完成，维度: (102717, 5)
    


```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 将稀疏技能矩阵转为 dense（注意：仅当内存允许时！10万×225 ≈ 22.5M元素，约180MB，可行）
skill_dense = skill_matrix.toarray()  # shape: (102717, 225)

# 2. 标准化结构化特征（虽然已归一化，但建议再标准化以适配后续模型）
scaler = StandardScaler()
struct_scaled = scaler.fit_transform(struct_features)  # shape: (102717, 5)

# 3. 拼接 → 统一特征矩阵（230维）
X_job = np.hstack([skill_dense, struct_scaled])  # shape: (102717, 230)
print("融合后岗位特征维度：", X_job.shape)
```

    融合后岗位特征维度： (102717, 230)
    


```python
def recommend_jobs(user_skills, user_city=None, user_category=None, 
                   user_edu=None, user_exp=None,  # 新增：学历和经验
                   min_salary=None, max_salary=None, 
                   required_skills=None,  # 新增：必须技能
                   min_skill_match=2,
                   top_k=10):
    """
    基于用户技能推荐岗位（改进版：硬性技能过滤 + 加权相似度）
    
    Parameters:
        user_skills: list, 用户技能列表
        user_city: str, 期望城市
        user_category: str, 期望职位类别
        user_edu: str, 学历（初中及以下/高中/大专/本科/硕士/博士）
        user_exp: int, 工作年限
        min_salary/max_salary: int, 薪资范围
        required_skills: list, 必须匹配的技能
        min_skill_match: int, 最少匹配技能数
        top_k: int, 推荐数量
    """
    
    # 1. 构建用户技能向量（225维）
    user_skill_vec = np.zeros(225)
    matched_skill_indices = []
    for skill in user_skills:
        skill_lower = skill.lower()
        for i, pattern in enumerate(unique_patterns):
            if skill_lower == pattern:
                user_skill_vec[i] = 1
                matched_skill_indices.append(i)
                break
    
    # 2. 构建用户结构化向量（5维）
    # 根据用户输入动态调整，不再用固定默认值
    edu_code = edu_order.get(user_edu, 4) if user_edu else 4  # 默认本科
    exp_code = user_exp if user_exp is not None else 3        # 默认3年
    
    user_struct = np.array([
        top_cities.index(user_city) / len(top_cities) if user_city in top_cities else 0.5,
        cat_map.get(user_category, 0) / len(cat_map) if user_category else 0.5,
        edu_code / 6,
        exp_code / 10,
        0.5  # 薪资分位数，中性值
    ]).reshape(1, -1)
    user_struct_scaled = scaler.transform(user_struct)
    
    # 3. 分别计算技能相似度和结构化相似度
    job_skill_part = X_job[:, :225]      # 所有岗位的技能部分
    job_struct_part = X_job[:, 225:]     # 所有岗位的结构化部分
    
    skill_sims = cosine_similarity(user_skill_vec.reshape(1, -1), job_skill_part)[0]
    struct_sims = cosine_similarity(user_struct_scaled, job_struct_part)[0]
    
    # 4. 计算每个岗位匹配了几个技能（硬性指标）
    match_counts = (user_skill_vec * job_skill_part).sum(axis=1)
    
    # 5. 硬性过滤条件
    mask = np.ones(len(df_clean), dtype=bool)
    
    # 技能硬性过滤：至少匹配 min_skill_match 个技能
    mask &= (match_counts >= min_skill_match)
    
    # 必须技能过滤（新增）
    if required_skills:
        for req in required_skills:
            req_lower = req.lower()
            req_col = None
            for i, p in enumerate(unique_patterns):
                if p == req_lower:
                    req_col = i
                    break
            if req_col is not None:
                mask &= (job_skill_part[:, req_col] == 1)
    
    # 学历过滤：岗位要求的学历 <= 用户学历+1（放宽一级）
    if user_edu:
        user_edu_level = edu_order.get(user_edu, 4)
        mask &= (df_clean['学历编码'] <= user_edu_level + 1).values
    
    # 经验过滤：岗位要求 <= 用户经验+2年
    if user_exp is not None:
        mask &= (df_clean['经验编码'] <= user_exp + 2).values
    
    # 城市/类别/薪资过滤
    if user_city:
        mask &= (df_clean['检索城市'] == user_city).values
    if user_category:
        mask &= (df_clean['检索二级职位类别'] == user_category).values
    if min_salary:
        mask &= (df_clean['平均薪资'] >= min_salary).values
    if max_salary:
        mask &= (df_clean['平均薪资'] <= max_salary).values
    
    # 6. 加权融合相似度（技能权重0.7，结构化0.3）
    similarities = 0.7 * skill_sims + 0.3 * struct_sims
    
    # 不符合硬性条件的设为-1（排除）
    similarities[~mask] = -1
    
    # 7. 取Top-K
    top_idx = np.argsort(similarities)[::-1][:top_k]
    
    # 8. 输出结果
    results = []
    for idx in top_idx:
        if similarities[idx] < 0:
            break
        
        row = df_clean.iloc[idx]
        
        # 计算具体匹配了哪些技能
        job_skills = set(np.where(job_skill_part[idx] == 1)[0])
        user_skills_set = set(matched_skill_indices)
        matched = [unique_patterns[i] for i in (job_skills & user_skills_set)]
        missing = [unique_patterns[i] for i in (job_skills - user_skills_set)]
        
        results.append({
            '岗位名': row['岗位名'],
            '公司': row['公司名称'],
            '城市': row['检索城市'],
            '类别': row['检索二级职位类别'],
            '薪资': f"{row['最小薪资']:.0f}-{row['最大薪资']:.0f}",
            '平均薪资': row['平均薪资'],
            '匹配度': similarities[idx],
            '匹配技能数': len(matched),
            '用户技能数': len(user_skills),
            '匹配技能': matched,
            '岗位额外要求': missing[:5]  # 只显示前5个未匹配技能
        })
    
    return pd.DataFrame(results)
```


```python
from ipywidgets import widgets, interact, Layout
from IPython.display import display, clear_output

# ==========================================
# Jupyter 交互式推荐系统
# ==========================================

def jupyter_recommend():
    """Jupyter Notebook 交互式界面"""
    
    # 创建输入控件
    skills_input = widgets.Textarea(
        value='python,mysql,docker,linux',
        placeholder='输入技能，用逗号分隔',
        description='技能:',
        layout=Layout(width='100%', height='60px')
    )
    
    city_input = widgets.Dropdown(
        options=[('不限', None)] + [(c, c) for c in top_cities],
        value='上海',
        description='城市:'
    )
    
    category_input = widgets.Dropdown(
        options=[('不限', None)] + [(c, c) for c in list(cat_map.keys())],
        value='后端开发',
        description='类别:'
    )
    
    edu_input = widgets.Dropdown(
        options=['不限', '初中及以下', '高中', '中技/中专', '大专', '本科', '硕士', '博士'],
        value='本科',
        description='学历:'
    )
    
    exp_input = widgets.IntSlider(
        value=3, min=0, max=10, step=1,
        description='经验(年):'
    )
    
    min_salary_input = widgets.IntText(
        value=15000, description='最低薪:'
    )
    
    max_salary_input = widgets.IntText(
        value=30000, description='最高薪:'
    )
    
    min_match_input = widgets.IntSlider(
        value=2, min=1, max=5, step=1,
        description='最少匹配技能:'
    )
    
    top_k_input = widgets.IntSlider(
        value=10, min=5, max=50, step=5,
        description='推荐数量:'
    )
    
    button = widgets.Button(
        description='🎯 开始推荐',
        button_style='primary',
        layout=Layout(width='200px', height='40px')
    )
    
    output = widgets.Output()
    
    # 布局
    form = widgets.VBox([
        widgets.HTML("<h2>🤖 IT岗位推荐系统</h2>"),
        skills_input,
        widgets.HBox([city_input, category_input]),
        widgets.HBox([edu_input, exp_input]),
        widgets.HBox([min_salary_input, max_salary_input]),
        widgets.HBox([min_match_input, top_k_input]),
        button,
        output
    ])
    
    display(form)
    
    # 按钮回调
    def on_button_click(b):
        with output:
            clear_output()
            
            # 获取参数
            user_skills = [s.strip() for s in skills_input.value.split(',') if s.strip()]
            user_city = city_input.value
            user_category = category_input.value
            user_edu = edu_input.value if edu_input.value != '不限' else None
            user_exp = exp_input.value
            min_salary = min_salary_input.value if min_salary_input.value > 0 else None
            max_salary = max_salary_input.value if max_salary_input.value > 0 else None
            min_skill_match = min_match_input.value
            top_k = top_k_input.value
            
            print(f"🔍 正在匹配: {user_skills} | {user_city} | {user_category}")
            
            # 执行推荐
            result = recommend_jobs(
                user_skills=user_skills,
                user_city=user_city,
                user_category=user_category,
                user_edu=user_edu,
                user_exp=user_exp,
                min_salary=min_salary,
                max_salary=max_salary,
                min_skill_match=min_skill_match,
                top_k=top_k
            )
            
            # 显示结果
            print(f"\n{'='*70}")
            print(f"🎯 找到 {len(result)} 个匹配岗位")
            print(f"{'='*70}")
            
            for i, row in result.iterrows():
                print(f"\n【{i+1}】 {row['岗位名']}")
                print(f"    🏢 {row['公司']} | 📍 {row['城市']}")
                print(f"    💰 {row['薪资']} | 📊 匹配度: {row['匹配度']:.4f}")
                print(f"    ✅ 匹配技能: {', '.join(row['匹配技能'])}")
                if row['岗位额外要求']:
                    print(f"    ⚠️  还需: {', '.join(row['岗位额外要求'])}")
                print("-" * 70)
    
    button.on_click(on_button_click)


# 在 Jupyter 里运行这个
jupyter_recommend()
```


    VBox(children=(HTML(value='<h2>🤖 IT岗位推荐系统</h2>'), Textarea(value='python,mysql,docker,linux', description='技能:…



```python

```
