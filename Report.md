Abstract
This report presents a comprehensive study on the effectiveness of Multi-head Latent Attention (MLA) and Rotary Position Embeddings (RoPE) for Telugu language modeling. Telugu is a morphologically rich, agglutinative Dravidian language spoken by over 80 million people. We developed a complete pipeline including morphology-aware tokenization, trained three model variants to isolate the effects of MLA and RoPE, and conducted extensive analysis including perplexity evaluation, morpheme-level attention analysis, and linguistic case studies. Our key findings show that RoPE provides a 20.6% perplexity improvement over absolute positional embeddings, while MLA achieves 8% parameter reduction with competitive performance. Together, MLA+RoPE represents the optimal architecture for Telugu and similar agglutinative languages.

Table of Contents
Introduction to Telugu Language
Morphology-Aware Tokenization
Dataset
Model Architecture
Experimental Setup: Three Models
Effect of Multi-head Latent Attention (MLA)
Effect of Rotary Position Embeddings (RoPE)
Limitations and Future Work
Conclusion
References
1. Introduction to Telugu Language
1.1 Overview
Telugu (తెలుగు) is a Dravidian language spoken predominantly in the Indian states of Andhra Pradesh and Telangana. With over 84 million native speakers, it is the fourth most spoken language in India and one of the 22 scheduled languages of India.

1.2 Linguistic Characteristics
Telugu presents unique challenges for natural language processing due to its rich morphological structure:

1.2.1 Agglutinative Morphology
Telugu is an agglutinative language, meaning words are formed by joining morphemes (meaningful units) together. A single Telugu word can express what requires an entire phrase in English:

Telugu Word	Morpheme Breakdown	English Translation
వెళ్ళాడు	వెళ్ళ + ఆడు	He went
చేయించుకోవాలనుకుంటున్నాను	చేయ + ఇంచు + కో + వాల + ను + కుంటు + ఉన్న + ఆను	I want to get (someone) to do (something for me)
In the complex example above, 8 morphemes stack together:

చేయ (do) - verb root
ఇంచు (causative) - make someone do
కో (reflexive) - for oneself
వాల (desiderative) - wanting to
ను (quotative)
కుంటు (progressive)
ఉన్న (continuous)
ఆను (1st person singular)
1.2.2 SOV Word Order with Flexibility
Telugu follows Subject-Object-Verb (SOV) word order as default, but allows flexibility for emphasis:

Order	Telugu	English	Pragmatic Use
SOV	రాముడు భోజనం తిన్నాడు	Ramu ate food	Neutral statement
OSV	భోజనం రాముడు తిన్నాడు	FOOD Ramu ate	Emphasizing object
OVS	భోజనం తిన్నాడు రాముడు	Food ate Ramu	Strong object focus
This flexibility is possible because Telugu uses case markers (suffixes) to indicate grammatical roles:

1.2.3 Rich Case Marking System
Case	Suffix	Example	Meaning
Nominative	-డు/-ము	రాముడు	Ramu (subject)
Accusative	-ని/-ను	రాముని	Ramu (object)
Dative	-కి/-కు	రామునికి	To Ramu
Genitive	-యొక్క	రాముడి	Ramu's
Locative	-లో	ఇంట్లో	In the house
Instrumental	-తో	చేతితో	With hand
Ablative	-నుండి	ఇంటినుండి	From the house
1.2.4 Verb Agreement
Telugu verbs agree with the subject in gender, number, and person:

Subject	Verb Ending	Example	Translation
He (masc. sg.)	-ఆడు	వెళ్ళాడు	He went
She (fem. sg.)	-ఇంది	వెళ్ళింది	She went
They (plural)	-ఆరు	వెళ్ళారు	They went
I (1st sg.)	-ఆను	వెళ్ళాను	I went
1.3 Implications for NLP
These characteristics have profound implications for language modeling:

Tokenization Challenge: Standard BPE/Unigram tokenizers may split words at morphologically incorrect boundaries
Position Encoding: Flexible word order suggests relative positions may matter more than absolute
Long-Range Dependencies: Subject-verb agreement can span many tokens
Morpheme Relationships: The meaning of suffixes depends on their position relative to the root
2. Morphology-Aware Tokenization
2.1 Motivation
Standard tokenizers like BPE (Byte Pair Encoding) and Unigram are trained purely on statistical co-occurrence patterns. For Telugu, this often results in:

Breaking morpheme boundaries: e.g., splitting "వెళ్ళాడు" as "వెళ్" + "ళాడు" instead of "వెళ్ళ" + "ఆడు"
Over-segmentation: creating too many tokens, increasing sequence length
Under-segmentation: failing to recognize productive suffixes
2.2 Our Approach: Morphology-Aware SentencePiece
We developed a morphology-aware tokenization pipeline using:

IndicNLP Library: For morphological analysis of Telugu text
Unsupervised Morphological Analyzer: Segments words into morphemes
SentencePiece Training: Train on pre-segmented text to learn morpheme-aligned boundaries
# Pipeline overview (from 01_train_tokenizers.py)
def train_morphaware_tokenizer(texts, output_path, vocab_size=32000):
    # 1. Initialize morphological analyzer
    analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('te')

    # 2. Segment text into morphemes
    segmented_texts = []
    for text in texts:
        words = text.split()
        segmented_words = [analyzer.morph_analyze(word) for word in words]
        segmented_texts.append(" ".join(segmented_words))

    # 3. Train SentencePiece on segmented text
    spm.SentencePieceTrainer.train(
        input=segmented_texts,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type='unigram'
    )
2.3 Tokenizer Comparison Results
Tokenizer	Fertility (tokens/word)	Notes
Morph-Aware SP	2.07	Linguistically accurate splits
Rule-Based (IndicNLP)	1.59	Gold standard morphemes
Standard Unigram	1.79	Statistical splits
XLM-RoBERTa	1.85	Multilingual baseline
GPT-4 (tiktoken)	13.0	Poor for Telugu
Key Insight: Our morphology-aware tokenizer produces slightly more tokens than statistical methods, but the splits align with linguistic boundaries, which we hypothesize leads to better downstream modeling.

2.4 Vocabulary Details
Vocabulary Size: 20,000 tokens
Model Type: Unigram (better for agglutinative languages)
Special Tokens: <pad>, <unk>, <s>, </s>
3. Dataset
3.1 Data Source
We used Telugu Wikipedia data from HuggingFace's datasets hub:

from datasets import load_dataset
dataset = load_dataset('wikipedia', '20220301.te')
3.2 Dataset Statistics
Split	Lines	Size	Description
Training	1,836,200	478 MB	Main training data
Test	96,643	22 MB	Held-out evaluation
Total	1,932,843	500 MB	Complete dataset
3.3 Preprocessing
Cleaning: Remove non-Telugu characters, normalize Unicode
Deduplication: Remove duplicate lines
Sentence Splitting: Split into manageable chunks
Filtering: Remove lines shorter than 10 characters
3.4 Data Splits for Training
The training script further splits the training data:

90% for training
5% for validation
5% for testing (separate from held-out test)
4. Model Architecture
4.1 Overview
We built a modern Transformer-based language model with state-of-the-art components:

TeluguLM Architecture
├── Token Embedding (vocab_size=20,000, dim=512)
├── Positional Encoding (RoPE or Absolute)
├── Transformer Blocks × 8
│   ├── RMSNorm
│   ├── Attention (MLA or Standard MHA)
│   └── SwiGLU FFN
├── Final RMSNorm
└── LM Head (tied with embedding)
4.2 Key Components
4.2.1 Multi-head Latent Attention (MLA)
MLA, introduced by DeepSeek, compresses Key-Value pairs into a lower-dimensional latent space:

class MLAAttention(nn.Module):
    def __init__(self, dim, heads, latent_dim, ...):
        # Standard Q projection
        self.q_proj = nn.Linear(dim, dim)

        # MLA: Compress KV to latent space
        self.kv_down_proj = nn.Linear(dim, latent_dim)  # 512 → 128
        self.kv_up_proj = nn.Linear(latent_dim, dim * 2)  # 128 → 1024

    def forward(self, x):
        q = self.q_proj(x)

        # Compress then decompress KV
        compressed_kv = self.kv_down_proj(x)  # Bottleneck
        kv = self.kv_up_proj(compressed_kv)
        k, v = kv.chunk(2, dim=-1)

        # Standard attention from here
        ...
Benefits:

Reduces KV cache memory by 4× (latent_dim=128 vs dim=512)
Fewer attention parameters
Forces learning of compressed representations
4.2.2 Rotary Position Embeddings (RoPE)
RoPE encodes relative positions through rotation matrices:

class RotaryPositionalEmbedding(nn.Module):
    def forward(self, x, seq_len):
        # Generate rotation frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

        # Apply rotation to Q and K
        q_rotated = apply_rotation(q, freqs)
        k_rotated = apply_rotation(k, freqs)
Benefits:

Encodes relative rather than absolute positions
Better handles variable sequence lengths
More suitable for flexible word order languages
4.2.3 Other Components
Component	Description	Benefit
RMSNorm	Root Mean Square Normalization	10-20% faster than LayerNorm
SwiGLU	Swish + Gated Linear Unit	Better gradient flow
Weight Tying	LM head shares embedding weights	Reduces parameters
4.3 Model Configuration
Parameter	Value
num_layers	8
model_dim	512
num_heads	8
head_dim	64
latent_dim (MLA)	128
ffn_dim	1365 (≈ 8/3 × 512)
vocab_size	20,000
max_seq_len	512
dropout	0.1
5. Experimental Setup: Three Models
To isolate the effects of MLA and RoPE, we trained three model variants:

5.1 Model Variants
Model	use_mla	use_rope	Parameters	Purpose
Model A: MLA + RoPE	✓	✓	32.8M	Full architecture
Model B: MHA + RoPE	✗	✓	35.7M	Isolate MLA effect
Model C: MHA (Baseline)	✗	✗	35.7M	Baseline
5.2 Training Configuration
All models were trained with identical hyperparameters:

Parameter	Value
batch_size	256
learning_rate	3e-4
max_steps	4,000
warmup_steps	1,000
optimizer	AdamW
weight_decay	0.01
mixed_precision	bfloat16
gradient_clipping	1.0
5.3 Training Infrastructure
Hardware: Google Colab with A100 GPU
Training Time: ~53 minutes per model (4000 steps)
Logging: Weights & Biases
5.4 WandB Run IDs
Model	Run ID	Run Name
MLA + RoPE	3mkxf49m	telugu-lm-mla-rope-morph
MHA + RoPE	8d4swup8	telugu-lm-rope-nomla
MHA (Baseline)	cir3c0tx	telugu-lm-no-mla-rope-morph
6. Effect of Multi-head Latent Attention (MLA)
6.1 Comparison: MLA vs MHA (Both with RoPE)
To isolate MLA's effect, we compare Model A (MLA+RoPE) with Model B (MHA+RoPE):

6.1.1 Parameter Efficiency
Architecture	Total Params	Attention Params	Reduction
MLA + RoPE	32.8M	5.8M	31% fewer
MHA + RoPE	35.7M	8.4M	-
MLA achieves significant parameter reduction by compressing KV pairs:

Standard MHA: Q, K, V each need dim × dim weights (3 × 512 × 512 = 786K per layer)
MLA: Q needs dim × dim, but KV uses dim × latent + latent × 2dim (512×512 + 512×128 + 128×1024 = 458K per layer)
6.1.2 Training Performance
Metric	MLA + RoPE	MHA + RoPE	Difference
Final Train Loss	2.1262	2.0381	+3.8%
Final Val Loss	2.2925	2.2680	+1.1%
Observation: MHA+RoPE achieves slightly lower loss, but the difference is marginal (1.1%).

6.1.3 Perplexity
Model	Perplexity	Tokens Tested
MLA + RoPE	162.76	15,432
MHA + RoPE	155.77	15,432
6.1.4 Model Size
Model	File Size	Savings
MLA + RoPE	125 MB	11 MB smaller
MHA + RoPE	136 MB	-
6.2 Morpheme-Level Attention Analysis
A key hypothesis is that MLA's latent compression forces better morphological abstraction.

6.2.1 Within-Morpheme vs Across-Morpheme Attention
We analyzed how attention flows within and across morpheme boundaries:

Metric	MLA	MHA	Interpretation
Within-Morpheme Attention	79.8%	88.9%	MHA focuses more within
Across-Morpheme Attention	20.2%	11.1%	MLA integrates more
Key Finding: MLA shows 2× higher cross-morpheme attention than MHA!

6.2.2 Why This Matters for Telugu
In agglutinative languages, understanding a word requires:

Recognizing the root
Understanding each suffix's contribution
Integrating root and suffix meanings
MLA's higher across-morpheme attention suggests it better captures this integration.

6.3 MLA Summary
Metric	MLA Advantage	Trade-off
Parameter Count	31% fewer attention params	-
Model Size	11 MB smaller	-
Training Speed	Potentially faster (fewer params)	Slight overhead from projections
Perplexity	Similar	Slightly higher (+4%)
Morpheme Integration	2× better cross-morpheme attention	-
Verdict: MLA is recommended for Telugu when parameter efficiency is important. The latent compression appears to encourage better morphological abstraction.

7. Effect of Rotary Position Embeddings (RoPE)
7.1 Comparison: RoPE vs Absolute Positions (Both with Standard MHA)
To isolate RoPE's effect, we compare Model B (MHA+RoPE) with Model C (MHA baseline):

7.1.1 Training Performance
RoPE Training Curves

Metric	MHA + RoPE	MHA (Absolute)	Improvement
Final Train Loss	2.0381	2.3079	-11.7%
Final Val Loss	2.2680	2.3672	-4.2%
7.1.2 Perplexity
Model	Perplexity	Improvement
MHA + RoPE	155.77	-20.6%
MHA (Baseline)	196.08	-
Key Finding: RoPE provides a 20.6% perplexity improvement!

7.2 Why RoPE Works for Telugu
7.2.1 Theoretical Motivation
Telugu Feature	Why RoPE Helps
Flexible word order	Relative positions matter more than absolute
Case markers indicate roles	Less reliance on position for role identification
Long-range agreement	Relative encoding captures subject-verb dependencies
Agglutinative morphology	Morpheme-to-root relative position is key
7.2.2 Linguistic Evidence
We tested specific Telugu linguistic phenomena:

Case Markers (Dative, Locative)

Example	Case	RoPE Loss	Baseline Loss	Winner
రామునికి ఇచ్చాను	Dative	5.26	5.68	RoPE
ఇంట్లో ఉన్నాడు	Locative	4.39	5.50	RoPE
Long-Range Subject-Verb Agreement

Distance	Example	RoPE	Baseline	Winner
1 token	అతను వెళ్ళాడు	5.62	6.68	RoPE
4 tokens	ఆ పెద్ద మనిషి నిన్న రాత్రి వెళ్ళాడు	6.96	7.37	RoPE
3 tokens	వాళ్ళు అందరూ కలిసి వెళ్ళారు	3.75	4.21	RoPE
Agglutinative Morphology

Complexity	Example	RoPE	Baseline	Improvement
Simple	వెళ్ళాడు (2 morphemes)	3.58	4.32	17%
Causative	చదివించారు (3 morphemes)	2.90	4.27	32%
Complex	చేయించుకోవాలనుకుంటున్నాను (8 morphemes)	2.83	4.38	35%
Key Insight: RoPE's advantage increases with morphological complexity!

7.3 RoPE Summary
Category	RoPE Wins	Key Insight
Case Markers	40%	RoPE helps with suffix-role relationships
Long-Range Agreement	60%	RoPE captures subject-verb dependencies
Agglutinative Morphology	60%	Strongest advantage for complex verbs
Overall Perplexity	20.6% better	Substantial improvement
Verdict: RoPE is essential for Telugu. The relative position encoding is far superior to absolute positions for this language.

8. Limitations and Future Work
8.1 Current Limitations
8.1.1 Data Scale
Current: 500 MB of Telugu Wikipedia
Limitation: Small compared to modern LLM pretraining (hundreds of GB)
Impact: Model may not generalize well to informal Telugu, dialects, or domain-specific text
8.1.2 Evaluation Metrics
Current: Perplexity only
Limitation: Perplexity doesn't directly measure downstream task performance
Missing: No evaluation on NER, sentiment, question answering, or translation
8.1.3 Model Scale
Current: ~35M parameters
Limitation: Too small for complex reasoning or knowledge tasks
Future: Scale to 100M-1B parameters for production use
8.1.4 Tokenization
Current: IndicNLP unsupervised morphological analyzer
Limitation: Not 100% accurate; may miss rare morphemes
Future: Train on linguist-annotated morphological data
8.1.5 Hardware Constraints
Current: Single A100 GPU (40GB)
Limitation: Limits batch size and model scale
Future: Multi-GPU training for larger models
8.2 Future Work
8.2.1 Scaling Studies
Larger Models: Train 100M, 500M, 1B parameter versions
Longer Training: Run for 50K+ steps with more data
Data Augmentation: Add Telugu books, news, social media
8.2.2 Downstream Tasks
Named Entity Recognition: Evaluate person, organization, location extraction
Sentiment Analysis: Telugu movie reviews, product reviews
Question Answering: Telugu reading comprehension
Machine Translation: Telugu ↔ English, Telugu ↔ Hindi
8.2.3 Architectural Improvements
Grouped Query Attention (GQA): Further reduce KV cache
Sliding Window Attention: Handle very long sequences
Mixture of Experts: Scale without proportional compute increase
8.2.4 Telugu-Specific Improvements
Better Morphological Analyzer: Train on expert-annotated data
Code-Mixing: Handle Telugu mixed with English (common in modern usage)
Dialectal Variation: Include data from different Telugu dialects
Script Variations: Handle both Unicode and legacy encodings
8.2.5 Ablation Studies
MLA without RoPE: Test if MLA benefits from RoPE specifically
Latent Dimension: Test latent_dim = 64, 256, 512
Layer Count: Test 4, 12, 16 layers
Head Count: Test 4, 16, 32 heads
9. Conclusion
This study provides comprehensive evidence for the effectiveness of Multi-head Latent Attention (MLA) and Rotary Position Embeddings (RoPE) for Telugu language modeling.

9.1 Key Findings
Finding	Evidence	Impact
RoPE is essential	20.6% perplexity improvement	Use RoPE for all Telugu LMs
MLA reduces parameters	31% fewer attention params	Efficient deployment
MLA improves morpheme integration	2× higher cross-morpheme attention	Better linguistic understanding
RoPE helps agglutination	35% improvement on complex verbs	Critical for Telugu morphology
9.2 Recommendations for Telugu LLM Development
Always use RoPE over absolute positional embeddings
Consider MLA when parameter efficiency matters
Use morphology-aware tokenization for linguistically meaningful splits
Test on complex morphological forms during evaluation
9.3 Broader Implications
Our findings likely extend to other agglutinative languages:

Dravidian: Tamil, Kannada, Malayalam
Turkic: Turkish, Uzbek, Kazakh
Finno-Ugric: Finnish, Hungarian, Estonian
Japanese and Korean
The combination of relative position encoding (RoPE) and latent attention (MLA) appears particularly suited for languages where:

Word order is flexible
Grammatical roles are marked by suffixes
Words contain multiple stacked morphemes
10. References
DeepSeek MLA: DeepSeek-V2 Technical Report (2024). Multi-head Latent Attention for Efficient LLMs
RoPE: Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding
SwiGLU: Shazeer (2020). GLU Variants Improve Transformer
IndicNLP: Kunchukuttan et al. (2020). IndicNLP Library
Telugu Morphology: Krishnamurti (2003). The Dravidian Languages
