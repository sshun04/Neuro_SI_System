treamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning
Shaolei Zhang1,3, Qingkai Fang1,3, Shoutao Guo1,3, Zhengrui Ma1,3, Min Zhang4, Yang Feng1,2,3*
1Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)
2Key Laboratory of AI Safety, Chinese Academy of Sciences 3University of Chinese Academy of Sciences, Beijing, China
4School of Future Science and Engineering, Soochow University zhangshaolei20z@ict.ac.cn, zhangminmt@hotmail.com, fengyang@ict.ac.cn
Abstract
Simultaneous speech-to-speech translation (Simul-S2ST, a.k.a streaming speech transla-tion) outputs target speech while receiving streaming speech inputs, which is critical for real-time communication. Beyond accomplish-ing translation between speech, Simul-S2ST requires a policy to control the model to gen-erate corresponding target speech at the oppor-tune moment within speech inputs, thereby pos-ing a double challenge of translation and pol-icy. In this paper, we propose StreamSpeech, a direct Simul-S2ST model that jointly learns translation and simultaneous policy in a unified framework of multi-task learning. Adhering to a multi-task learning approach, StreamSpeech can perform offline and simultaneous speech recognition, speech translation and speech syn-thesis via an “All-in-One” seamless model. Ex-periments on CVSS benchmark demonstrate that StreamSpeech achieves state-of-the-art per-formance in both offline S2ST and Simul-S2ST tasks. Besides, StreamSpeech is able to present high-quality intermediate results (i.e., ASR or translation results) during simultaneous trans-lation process, offering a more comprehensive real-time communication experience1.
1 Introduction
Simultaneous speech-to-speech translation (Simul-S2ST), which involves generating target speech while concurrently receiving streaming speech in-puts (Salesky et al., 2023; OpenAI, 2024), has be-come an indispensable technology for low-latency communication in various scenarios, such as in-ternational conferences, live broadcasts and on-line subtitles. To produce high-quality translated speech under low latency, Simul-S2ST requires a policy to determine the optimal moments to start translating within the streaming speech inputs (i.e.,
*Corresponding author: Yang Feng 1Code: https://github.com/ictnlp/StreamSpeech
Project Page: https://ictnlp.github.io/StreamSpeech-site/
Streaming Inputs
bonjour à tous …Streaming ASR:
hello everyone …Simul-S2TT:
Simul-S2ST:
StreamSpeech
“All in One” seamless model for offline and simultaneous ASR / Translation / Synthesis
Simultaneous Outputs
Figure 1: StreamSpeech is an “All in One” seamless model for multiple offline and simultaneous tasks.
READ action) and subsequently generate coher-ent target speech outputs (i.e., WRITE action) (Gu et al., 2017).
Existing simultaneous translation methods focus on text-to-text (Simul-T2TT) (Ma et al., 2019; Ari-vazhagan et al., 2019; Zhang and Feng, 2023b) and speech-to-text translation (Simul-S2TT) (Ren et al., 2020; Chen et al., 2021; Zeng et al., 2021; Zhang and Feng, 2023a). Such methods typically require cascading external modules such as speech recog-nition (ASR) and text-to-speech synthesis (TTS) to accomplish Simul-S2ST. However, this cascaded approach tends to amplify inference errors progres-sively between modules (Zhang et al., 2022a; Ma et al., 2020b), and also impedes the joint optimiza-tion of various modules (Zhang and Feng, 2023c). To address these issues, developing a direct Simul-S2ST model is imperative, particularly given the promising potential exhibited by end-to-end mod-els such as SeamlessM4T (Communication et al., 2023b) and GPT-4o (OpenAI, 2024).
Direct speech-to-speech translation (S2ST) is already highly challenging, and the goal of accom-plishing it simultaneously (Simul-S2ST) further
8964
Streaming Speech Encoder Simultaneous Text Decoder
Text-to-Unit Encoder
Unit CTC Decoder
: _hello _every one
: 63 16 3 23 23 54
…Source CTC Decoder Target CTC Decoder
: _hello _every one : _bonjour _à _tous
ASR NAR-S2TT
AR-S2TT
S2UT
… … …
<s> _hello _every one
READ or not? WRITE or not? HiFi-GAN Vocoder
Figure 2: StreamSpeech employs two-pass architecture that first converts source speech into target text hidden states Dtext (autoregressive speech-to-text translation, AR-S2TT) and then generates target speech via non-autoregressive text-to-unit generation. The source/target/unit CTC decoders are introduced to learn alignments via multiple tasks of speech recognition (ASR), non-autoregressive speech-to-text translation (NAR-S2TT) and speech-to-unit translation (S2UT), accordingly guiding StreamSpeech when to start recognizing, translating and synthesizing.
exacerbates the difficulty. For translation, speech involves more diverse representation due to addi-tional features such as timbre and intonation (Jia et al., 2022a), which renders directly translating source speech to target speech challenging. In simultaneous scenarios, beyond translation, the model additionally requires a policy to identify the appropriate translating moments, which is non-trivial to directly accomplish due to the continuous nature and uncertain duration of speech (Zhang and Feng, 2023a). Therefore, Simul-S2ST faces the double challenges of translation and policy.
To address the challenges of translation and pol-icy, we aim to introduce textual information of both source and target speech to guide Simul-S2ST, which can not only provide intermediate supervi-sion for translation but also guide the policy by establishing an alignment between source and tar-get speech with text as a bridge. Specifically, a reasonable policy should control the model to wait until recognizing the presence of text in the re-ceived speech (READ), facilitated by the alignment between source speech and source text. Subse-quently, the model should generate target speech corresponding to inputs (WRITE), which can be guided by the alignments from the source speech to target text and from the target text to target speech.
Given the pivotal role of text in both translation and alignment-guided policy, we propose Stream-Speech, a direct Simul-S2ST model that jointly learns translation and policy in a unified framework of multi-task learning. StreamSpeech employs the advanced two-pass architecture (Inaguma et al.,
2023; Jia et al., 2022a), which first translates source speech into target text hidden states, and then con-verts the text hidden states into target speech. Fur-thermore, we introduce multiple connectionist tem-poral classification (CTC) (Graves et al., 2006) de-coders and optimize them via auxiliary tasks of ASR and S2TT, thereby providing intermediate supervision for translation and meanwhile learn-ing alignments to guide policy. All modules in StreamSpeech are jointly optimized through multi-task learning, facilitating jointly learning of trans-lation and policy. Experiments show that Stream-Speech exhibits adaptability to different latency and achieves state-of-the-art performance on both offline S2ST and Simul-S2ST tasks.
2 Background
Speech-to-Speech Translation (S2ST) The cor-pus we used for speech-to-speech translation (S2ST) task is denoted as quadruple: D = {(X,A, Y, S)}, where X =
( x1, · · · , x|X|
) is the
source speech, A = ( a1, · · · , a|A|
) is the tran-
scribed text of source speech, Y = ( y1, · · · , y|Y |
)
is the target text, S = ( s1, · · · , s|S|
) is the target
speech. The current mainstream methods for S2ST (Inaguma et al., 2023) extract a discrete unit se-quence U=
( u1, · · · , u|U |
) from the target speech,
and employ a two-pass architecture, where both the first and second passes use autoregressive encoder-decoder. The first pass transforms the source speech to target text hidden states, and the sec-ond pass generates the discrete unit sequence based on the text hidden states, followed by a pre-trained
8965
unit-based HiFi-GAN vocoder (Kong et al., 2020) for target speech synthesis. In addition to the pri-mary speech-to-unit translation (S2UT, X → U ), an auxiliary speech-to-text translation task (S2TT, X → Y ) is introduced to provide supervision.
Connectionist Temporal Classification (CTC) (Graves et al., 2006) CTC is a technique used to model alignment between two sequences of un-equal lengths. For a longer input sequence X , CTC decoder generates a same-length sequence Z con-taining repeated and blank tokens ϕ, which is sub-sequently shortened by merging consecutively re-peated tokens and removing blank tokens ϕ via collapsing function Π(·). During training, given the ground-truth sequence Y , CTC loss is calcu-lated on all sequences Z that can be reduced to Y via the collapsing function:
CTC(X ,Y) = − log ∑
Z∈Π−1(Y)
p (Z | X ) . (1)
3 StreamSpeech
3.1 Architecture
The overall architecture of StreamSpeech is illus-trated in Figure 2. StreamSpeech consists of three parts: streaming speech encoder, simultaneous text decoder and synchronized text-to-unit generation module. Multiple CTC decoders are introduced to learn the alignments through auxiliary tasks and accordingly guide the policy.
Streaming Speech Encoder Conformer archi-tecture exhibits remarkable advantages in speech modeling by stacking attention modules and con-volutional modules (Gulati et al., 2020), while it struggles to model the streaming speech inputs, pri-marily due to the bi-directional self-attention and convolutional operations involving the entire se-quence’s receptive field. To this end, we propose chunk-based Conformer, aiming to endow the Con-former architecture with the capability to encode streaming inputs while retaining the bi-directional encoding within local chunk.
Figure 3 shows the architecture of chunk-based Conformer. First of all, the raw speech inputs are converted to speech features (we use filterbank fea-tures (Povey et al., 2011) in our work), where each speech feature typically corresponds to a 40ms du-ration. Chunk-based Conformer divides the stream-ing speech into chunks, each containing C speech features, where C is a hyperparameter controlling the chunk size. In the chunk-based Conformer,
Chunk-based Self-attention Module
Feed Forward
Chunk-based Convolution Module
Feed Forward convolution
Q ue ry
Key
attention
mask
padding
×N
Figure 3: Architecture of chunk-based Conformer.
self-attention and convolution operations are both bidirectional within a chunk and unidirectional be-tween chunks, thereby handling the streaming in-puts. For chunk-based self-attention, feature xi pays attention to the features xj that are located in the same and previous chunks, calculated as:
ChunkAttn (xi, xj)
=
{ Attn (xi, xj) if j ≤
⌈ i C
⌉ × C
0 otherwise ,
(2)
where Attn (xi, xj) is standard multi-head atten-tion (Vaswani et al., 2017), and ⌈·⌉ is ceiling opera-tion. For chunk-based convolution, the convolution operation with kernel size k is truncated at the up-per bound of the chunk, calculated as:
ChunkConv (xi) = (3)
Conv ( xi− k−1
2 ,· · ·, xi,· · ·, xmin(i+ k−1
2 ,⌈ i
C⌉×C)
) .
where ⌈
i C
⌉ × C is the upper bound of the chunk
that xi is located in. In implementation, chunk-based convolution can be computed in parallel through a mask operation (mask out those trun-cated positions). Through the streaming encoder, the source speech hidden states are calculated, de-noted as H =
( h1, · · · , h|H|
) . With chunk-based
Conformer, the streaming speech encoder not only fulfills the need for streaming encoding but also conducts local bi-directional encoding of speech.
Simultaneous Text Decoder After streaming encoder, text decoder simultaneously generates tar-get text Y by attending the source speech hidden states H . To achieve this, StreamSpeech requires a policy to decide when to generate each target to-ken (i.e., how many speech states can the decoder attend to.). A reasonable policy should ensure that the model waits until recognizing the source text in the source speech (READ), and then generates the corresponding target text (WRITE).
8966
To this end, we aim to align the source and target text to the speech inputs, thereby guiding “READ or not” and “WRITE or not” respectively. Con-sidering the length difference between speech and text sequences, we align them via CTC decoder (refer to Sec.2). Specifically, we introduce a source CTC decoder CTCDecA(·) and a target CTC decoder CTCDecY(·) at the top of the streaming speech en-coder to generate source and target text:
Dasr = CTCDecA(H), (4)
Dnar-s2tt = CTCDecY(H), (5)
and optimize them through the auxiliary tasks of speech recognition (ASR, X → A) and non-autoregressive speech-to-text translation (NAR-S2TT, X → Y ), via CTC loss respectively:
Lasr = CTC(Dasr, A), (6)
Lnar-s2tt = CTC(Dnar-s2tt, Y ). (7)
With CTC decoders, the source and target text are aligned to source speech. Accordingly, Stream-Speech starts translating upon the source CTC de-coder recognizing a new source token from source speech, and then autoregressively generates target tokens that align to the received speech within tar-get CTC decoder2. Therefore, we calculate the number of source tokens and target tokens aligned to the current speech inputs X≤j , denoted as N asr
j
and N nar-s2t j , respectively. Note that during train-
ing, we calculate the expected number of tokens contained in the CTC sequence, where the specific calculation is introduced in Appendix A.
Given N asr j and N nar-s2tt
j , StreamSpeech au-toregressively generates target token yi after receiv-ing speech X≤g(i), where g (i) is defined as:
g (i) = argmin {j | Nasr
j−1<Nasr j }
( N nar-s2tt
j ≥ i ) . (8)
N asr j−1 < N asr
j ensures that StreamSpeech starts translating when a new source token is recognized, and (N nar-s2tt
j ≥ i) ensures that StreamSpeech generates those target tokens that align to the re-ceived speech. Based on the policy guided by the alignments derived from ASR and NAR-S2TT, si-multaneous text decoder generates yi after receiv-ing speech X≤g(i), and optimized via cross-entropy
2NAR-S2TT can achieve well 1-gram token accuracy, but its translations are often less smooth compared to AR-S2TT. Therefore, StreamSpeech adopts NAR-S2TT to capture align-ment and guide the policy, while still leveraging AR-S2TT to generate target tokens for better translation quality.
loss on autoregressive speech-to-text translation (AR-S2TT, X → Y ):
Lar-s2tt=− 1
|Y |
|Y |∑
i=1
log p ( yi | X≤g(i), Y<i
) . (9)
Non-autoregressive Text-to-Unit Generation To synchronously generate the corresponding unit for the currently generated text, StreamSpeech em-ploys a non-autoregressive text-to-unit (T2U) archi-tecture (Gu et al., 2018), comprising a T2U encoder and a unit CTC decoder. T2U encoder takes the hid-den state Dtext from the simultaneous text decoder as inputs. For the unit CTC decoder, considering that unit sequences U are often longer than text se-quences Y , we upsample the T2U encoder outputs r times as the decoder inputs, where the ith input corresponds to Dtext
⌈i/r⌉. Then unit CTC decoder gen-erates the unit sequence U non-autoregressively by attending to those T2U encoder outputs located before Dtext
⌈i/r⌉. Formally, the output Dunit of unit CTC decoder CTCDecU is calculated as:
Dunit i = CTCDecU
( Dtext
≤⌈i/r⌉ ) . (10)
NAR T2U generation is optimized on speech-to-unit translation task (S2UT, S → U ) via CTC loss:
Ls2ut = CTC(Dunit, U). (11)
Finally, a unit-based HiFi-GAN vocoder (Kong et al., 2020) is used to synthesize target speech based on the generated units. Note that the HiFi-GAN vocoder is often pre-trained and frozen.
3.2 Training All tasks involved in StreamSpeech are jointly op-timized via multi-task learning in an end-to-end manner, and the total training objective L encom-passes the losses of S2UT, AR-S2TT, ASR, and NAR-S2TT tasks:
L = Ls2ut + Lar-s2tt + Lasr + Lnar-s2tt. (12)
Multi-task learning effectively integrates the learn-ing of simultaneous policy and translation into a unify framework. Besides, the high-quality inter-mediate results of auxiliary tasks, such as ASR and AR-S2TT, can also be displayed to users during inference as supplementary products.
Multi-chunk Training During inference, Simul-S2ST may face different latency requirements, and training multiple models for every latency is expen-sive (Elbayad et al., 2020; Zhang and Feng, 2021b).
8967
Algorithm 1: Inference of StreamSpeech Input :streaming speech inputs X , chunk size C,
current received speech X̂ Output : target speech outputs S
1 while |X̂| ≤ |X| do 2 generate ASR results Â, with Eq.(5); 3 generate NAR-S2TT results Ŷ , with Eq.(5); 4 if |Â|> |A| and |Ŷ |> |Y | then // WRITE
5 A = Â; 6 while |Y | < |Ŷ | and Y−1 ̸= <eos> do 7 generate target token y, with Eq.(9); 8 Y.append(y) 9 end
10 generate units U of Y , with Eq.(10); 11 S = Vocoder(U); 12 // output new generated speech 13 else // READ 14 wait for next speech chunk; 15 X̂.append(X|X̂|:|X̂|+C); 16 end 17 end
To this end, we introduce multi-chunk training to improve the performance of StreamSpeech across various latency levels. In multi-chunk training, chunk size C of streaming speech encoder is not fixed, but randomly sampled from a uniform distri-bution for 1 to |X|, expressed as C ∼ U (1, |X|), where c = |X| refers to offline S2ST. With multi-chunk training, a single StreamSpeech model can cater to different latency requirements.
3.3 Inference
Algorithm 1 illustrates the inference policy of StreamSpeech. During inference, StreamSpeech processes streaming speech inputs based on the set chunk size C, where each speech feature typically corresponds to 40ms duration (e.g., C = 8 means encoding speech inputs every C × 40 = 320ms). Then StreamSpeech decodes the source tokens Â and target tokens Ŷ associated with the currently received speech X̂ through the CTC decoders for ASR and NAR-S2TT tasks. In cases where new source token is recognized, and the count of aligned target tokens surpasses the previously generated target tokens (line 5), StreamSpeech autoregres-sively generates the target tokens (line 7-10) and synchronously generates the corresponding units (line 11) and synthesizes the target speech (line 12); otherwise StreamSpeech waits for the next speech chunk of size C. Due to the proposed multi-chunk training, StreamSpeech can control the latency by adjusting chunk size C during inference, where the smaller C will lead to lower latency.
4 Experiments
4.1 Experimental Setup Datasets We conduct experiments on CVSS-C benchmark (Jia et al., 2022b), which is a large-scale S2ST corpus derived from the CoVoST 2 speech-to-text translation corpus (Wang et al., 2020) by syn-thesizing the target speech using a state-of-the-art TTS system. We evaluate StreamSpeech on CVSS-C French→English (Fr→En), Spanish→English (Es→En) and German→English (De→En).
Pre-processing Following Inaguma et al. (2023), we convert the source speech to 16000Hz and generate target speech with 22050Hz. For source speech, we compute 80-dimensional mel-filterbank features (Povey et al., 2011) and apply global-level cepstral mean-variance normalization, where each speech feature corresponds to 40ms duration. For target speech, we extract the discrete units via mHuBERT3 (Popuri et al., 2022), and synthesize target speech through a pre-trained unit-based HiFi-GAN vocoder4 (Kong et al., 2020). For source and target text, we use SentencePiece (Kudo and Richardson, 2018) to generate a unigram vo-cabulary of size 6000, respectively.
4.2 Systems Settings Since StreamSpeech can be applied to simultane-ous and offline S2ST, we compare StreamSpeech with offline S2ST and Simul-S2ST models.
Offline S2ST baselines include S2UT (Lee et al., 2022), Translatotron (Jia et al., 2019), Transla-totron 2 (Jia et al., 2022a), DASpeech (Fang et al., 2023) and UnitY (Inaguma et al., 2023), where UnitY is the state-of-the-art offline S2ST model and is the basic framework of seamlessM4T (Com-munication et al., 2023a). Refer to Appendix C for detailed introduction to these offline baselines.
Simul-S2ST baselines include: Wait-k (Ma et al., 2019) Wait-k policy first waits
for k×320ms of speech, and then generates a target word every 320ms (Ma et al., 2020b). We apply wait-k policy on UnitY, where the first pass adopts wait-k policy, and then the second pass generates units until <eos> token.
ASR+HMT+TTS (cascaded) (Zhang and Feng, 2023b) Hidden Markov Transformer5 (HMT) is
3https://dl.fbaipublicfiles.com/hubert/ mhubert_base_vp_en_es_fr_it3.pt
4https://dl.fbaipublicfiles.com/fairseq/ speech_to_speech/vocoder/code_hifigan/mhubert_ vp_en_es_fr_it3_400k_layer11_km1000_lj
5https://github.com/ictnlp/HMT
8968
Models #Param. Fr→En Es→En De→En Average greedy beam10 greedy beam10 greedy beam10 greedy beam10
Ground Truth - 84.52 88.54 75.53 82.86
S2UT 73M 20.91 22.23 16.94 18.53 2.46 2.99 13.44 14.58 Translatotron 79M 16.96 / 8.72 / 1.97 / 9.22 / Translatotron 2 87M 25.49 26.07 22.35 22.93 16.24 16.91 21.36 21.97 DASpeech 93M 25.03 / 21.37 / 16.14 / 20.85 / UnitY 67M 26.90 27.77 23.93 24.95 18.19 18.74 23.01 23.82
StreamSpeech 70M 27.58∗∗ 28.45∗∗ 26.16∗∗ 27.25∗∗ 19.72∗∗ 20.93∗∗ 24.49 25.54
Table 1: Offline S2TT results (ASR-BLEU) on CVSS-C Fr→En, Es→En, De→En test sets. We report the results under greedy and beam=10 decoding, where Translatotron only supports greedy decoding and DASpeech uses Viterbi decoding. ∗∗ means the improvements over the SOTA UnitY are statistically significant (p < 0.01).
the state-of-the-art simultaneous text-to-text trans-lation model. We train the streaming ASR and real-time TTS model and add them before and after HMT to form a cascaded Simul-S2ST system.
DiSeg+TTS (cascaded) (Zhang and Feng, 2023a) Differentiable segmentation6 (DiSeg) is the state-of-the-art simultaneous speech-to-text transla-tion model. We also add real-time TTS model after DiSeg to form a cascaded Simul-S2ST system.
StreamSpeech Our direct Simul-S2ST model. All implementations are adapted from Fairseq
Library (Ott et al., 2019). StreamSpeech uses ba-sically the same settings as UnitY (Inaguma et al., 2023), and the introduced CTC decoder consists of only a fully connected layer. Other model con-figurations and training details are reported in Ap-pendix H. The only hyperparameter that needs to be set in StreamSpeech is the upsampling rate r in NAR T2U generation, where we set r = 25 based on validation in Appendix D. For cascaded systems, the streaming ASR and real-time TTS modules use the streaming encoder and non-autoregressive text-to-unit module, identical to those used in Stream-Speech, for a fair comparison.
4.3 Evaluation We apply SimulEval7 (Ma et al., 2020a) to evaluate the Simul-S2ST from both quality and latency.
Quality We evaluate S2ST quality using ASR-BLEU toolkit8, which first transcribes the trans-lated speech into text using a pre-trained ASR model and then calculates the SacreBLEU (Post, 2018) score with reference. We also use BLASER 2.0 to assess the generated speech’s quality and the results are reported in Appendix E and J.
6https://github.com/ictnlp/DiSeg 7https://github.com/facebookresearch/SimulEval 8https://github.com/facebookresearch/fairseq/
tree/ust/examples/speech_to_speech/asr_bleu
Models Fr→En Es→En De→En ASR-BLEU Speedup ASR-
BLEU Speedup ASR-BLEU Speedup
UnitY 27.77 1.0× 24.95 1.0× 18.74 1.0× StreamSpeech 28.45 3.6× 27.25 4.5× 20.93 4.5×
Table 2: Speedup of StreamSpeech.
Latency We use Average Lagging (AL) (Ma et al., 2020b) to evaluate the latency, where AL measures the average duration (ms) that outputs lag behind inputs. We also measure the computation-aware latency, which includes the computational time of the model. The computation-aware latency is evaluated on 1 NVIDIA RTX 3090 GPU with batch-size=1. More latency metrics are reported in Appendix I to show latency performance.
4.4 Main Results We conduct experiments in both offline S2ST and Simul-S2ST tasks.
Offline S2ST Table 1 reports the performance of StreamSpeech in offline S2ST, where Stream-Speech outperforms the state-of-the-art UnitY with an average improvement of 1.5 BLEU. Stream-Speech uses two-pass architecture and achieves significant improvements over S2UT and Transla-totron, which use one-pass architecture. For two-pass architecture, DASpeech employs NAR archi-tecture in both first and second passes (Fang et al., 2023), while UnitY uses AR architecture in two passes (Inaguma et al., 2023). StreamSpeech uses AR architecture in S2TT task (first pass) that in-volves more reordering and context dependence, and NAR architecture in T2U task (second pass) that is basically monotonically aligned. This effec-tively mitigates the impact of the NAR architecture on modeling capabilities and meanwhile captures the alignment between text and unit. Overall, multi-
8969
1000 2000 3000 4000 5000 6000 Average Lagging (AL, ms)
12
14
16
18
20
22
24
26
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech StreamSpeech (computation aware) Wait-k Wait-k (computation aware)
(a) Fr→En
1000 2000 3000 4000 5000 6000 7000 Average Lagging (AL, ms)
6
8
10
12
14
16
18
20
22
24
26
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech StreamSpeech (computation aware) Wait-k Wait-k (computation aware)
(b) Es→En
1000 2000 3000 4000 5000 6000 Average Lagging (AL, ms)
6
8
10
12
14
16
18
20
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech StreamSpeech (computation aware) Wait-k Wait-k (computation aware)
(c) De→En
Figure 4: Simul-S2ST results (quality against latency) on CVSS-C Fr→En, Es→En, De→En test sets. The hollow points represent computation-aware latency, which includes the inference time consumed by the model. Some simultaneous outputs of StreamSpeech can be heard at https://ictnlp.github.io/StreamSpeech-site/.
Models Tasks ASR NAR-S2TT AR-S2TT S2UT S2ST ASR NAR-S2TT AR-S2TT S2UT WER↓ BLEU↑ ACC↑ BLEU↑ ACC↑ BLEU↑ ASR-BLEU↑
UnitY ✘ ✘ ✔ ✔ / / 31.31 61.0 33.47 27.77
StreamSpeech
✘ ✘ ✔ ✔ / / 31.20 61.5 31.37 27.47 ✘ ✔ ✔ ✔ / 22.95 59.9 31.56 61.1 31.15 27.73 ✔ ✘ ✔ ✔ 20.70 / 32.28 62.3 31.42 28.18 ✔ ✔ ✔ ✔ 20.55 23.82 60.9 32.60 62.4 31.72 28.45
Table 3: Ablation study of multi-task learning on offline S2ST, evaluated on CVSS-C Fr→En test set. We report word error rate (WER) for ASR task, BLEU score and 1-gram accuracy (ACC) for NAR-S2TT and AR-S2TT tasks, BLEU score (computes on unit sequences) for S2UT task, and ASR-BLEU score for S2ST task.
task learning not only guides the policy, but also provides intermediate supervision for translation, yielding superior offline S2ST performance.
Speedup of StreamSpeech To explore the in-ference efficiency of StreamSpeech, we report the speedup of StreamSpeech compared to UnitY in Ta-ble 2. In the two-pass architecture, StreamSpeech employs an autoregressive structure in the first pass for translation and a non-autoregressive structure in the second pass for speech synthesis (where the sequences are longer but monotonically aligned). This AR+NAR two-pass architecture brings signifi-cant speedup while maintaining translation quality.
Simul-S2ST Figure 4 shows the Simul-S2ST performance of StreamSpeech, where Stream-Speech outperforms Wait-k under all latency, par-ticularly exhibiting a roughly 10 BLEU improve-ment under low latency. Wait-k stands as the most widely used policy and achieves good performance on simultaneous T2TT and S2TT (Ma et al., 2019, 2020b). For the Simul-S2ST task where the source and target sequences are both continuous speech, StreamSpeech’s policy derived from alignments enables the model to translate at more appropri-ate moments and generate coherent target speech,
resulting in significant advantages. Moreover, con-cerning computation-aware latency, StreamSpeech introduces only a marginal increase in parameters, thus avoiding notable inference overhead.
Direct Simul-S2ST v.s. Cascaded Simul-S2ST Figure 5 presents a comparison between the di-rect and cascaded Simul-S2ST models, evaluated on the CVSS-C Fr→En test set. The results sug-gest a general superiority of the direct model over the cascaded systems. Specifically, when we de-compose the direct StreamSpeech into two mod-ules “S2TT+TTS”, error accumulation leads to a 1 BLEU decrease under low latency, even with the same policy. Furthermore, compared to the cas-caded system comprising state-of-the-art HMT and DiSeg, StreamSpeech demonstrates a significant advantage, underscoring the superiority of direct StreamSpeech in Simul-S2ST task.
5 Analyses
5.1 Effect of Multi-task Learning StreamSpeech jointly optimizes S2UT, AR-S2TT, ASR, and NAR-S2TT tasks through multi-task learning. To verify the effect of multi-task learning, we conduct an ablation study of auxiliary tasks on
8970
1000 2000 3000 4000 5000 Average Lagging (AL, ms)
12
14
16
18
20
22
24
26
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech (direct) StreamSpeech (S2TT + TTS, cascaded) DiSeg + TTS (cascaded) ASR + HMT + TTS (cascaded) Wait-k (direct)
Figure 5: Comparison of direct and cascaded Simul-S2ST systems.
1000 2000 3000 4000 5000 Average Lagging (AL, ms)
12
14
16
18
20
22
24
26
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech StreamSpeech (w/o multi-chunk training) Wait-k
Figure 6: Effect of multi-chunk train-ing in StreamSpeech.
1000 2000 3000 4000 5000 Average Lagging (AL, ms)
12
14
16
18
20
22
24
26
A SR
-B LE
U
StreamSpeech (Offline) UnitY (Offline) StreamSpeech StreamSpeech (w/o Source Speech Source Text) StreamSpeech (w/o Source Speech Target Text) StreamSpeech (w/o Target Text Target Speech) Wait-k
Figure 7: Ablation study on align-ments in policy.
Train \ Test C=8 C=16 C=32 C=64 C=∞ C = 8 24.91 24.72 25.03 24.82 23.37 C = 16 24.18 25.64 25.75 25.62 24.76 C = 32 23.06 24.69 25.82 25.85 25.75 C = 64 19.55 22.77 24.63 25.94 26.41 C = ∞ 1.42 7.12 14.58 21.76 26.90
Multi-Chunk 25.34 25.97 26.31 26.61 26.47
Table 4: Offline S2ST results on various chunk size C of streaming encoder during training and testing.
CVSS-C Fr→En offline S2ST. As reported in Table 3, the introduction of aux-
iliary tasks effectively improves the performance of S2ST. Multi-task learning offers staged interme-diate supervision for each module within Stream-Speech (Tang et al., 2021a,b), thereby enhancing overall performance. Furthermore, it is noteworthy that NAR-S2TT exhibits a notable gap in BLEU scores compared to AR-S2TT, while the 1-gram ac-curacy shows minimal differences. This highlights the rationale behind utilizing NAR-S2TT for align-ing source speech and target text and employing AR-S2TT for translation.
5.2 Superiority of Multi-chunk Training
To enhance the performance of StreamSpeech un-der various latency, we propose multi-chunk train-ing. Figure 6 illustrates the Simul-S2ST perfor-mance on Fr→En test set when employing multi-chunk training and training multiple separate mod-els for different latency. The results indicate that multi-chunk training performs better under all la-tency. More importantly, multi-chunk training en-ables StreamSpeech to handle Simul-S2ST under various latency conditions using just one model.
Chunk-based Conformer To further evaluate the impact of multi-chunk training on the modeling capabilities of chunk-based Conformer, we conduct
experiments by training StreamSpeech with vari-ous chunk sizes C and testing them with different test chunk sizes in offline S2ST. The results are reported in Table 4, evaluated on CVSS-C Fr→En test set. The results indicate that models trained with a single chunk size often excel only at a par-ticular test chunk size and struggle to adapt to oth-ers. Multi-chunk training equips StreamSpeech with the capability to adapt to different chunk sizes, thereby enabling it to handle S2ST under various latency conditions using a single model. Notably, multi-chunk training also demonstrates superior performance at smaller chunk sizes, which in line with previous findings suggesting that incorporat-ing future information during training has a positive improvement (Ma et al., 2019; Zhang et al., 2021; Zhang and Feng, 2022d; Guo et al., 2024b).
5.3 Analysis on StreamSpeech Policy
StreamSpeech models alignments between source speech and source/target text, target text and target speech, thereby allowing for the adaptive decision of READ/WRITE actions. To assess the signif-icance of these three alignments, we present the CVSS-C Fr→En performance when removing one of them (refer to Appendix B for detailed introduc-tion of ablation settings) in Figure 7.
The results underscore the pivotal role of mod-eling the alignment between target text and target speech through NAR text-to-unit module, as the number of units corresponding to text is often di-verse, and the proposed unit CTC decoder effec-tively addresses this issue. Besides, the alignment between source speech and source/target text en-sures StreamSpeech starts translating at appropriate moments and generates a reasonable number of tar-get tokens, where removing any of these alignment components results in performance degradation.
8971
Models #Parm. AL (ms)↓ WER↓ Wav2Vec2-large 315M 5684.38 26.17 Whisper-base 74M 5684.38 38.04
StreamSpeech 70M (33M used)
109.127 25.46 267.891 25.54 431.652 25.20 757.989 24.67
Table 5: Streaming ASR results on Fr→En test set.
0 1000 2000 3000 4000 5000 6000 Average Lagging (AL, ms)
22
24
26
28
30
B LE
U
UnitY (Offline) StreamSpeech DiSeg ASR + HMT Wait-k
Figure 8: Simultaneous speech-to-text translation re-sults on Fr→En test set.
Totally, the alignments involved in StreamSpeech are reasonable and can be jointly trained with trans-lation through multi-task learning.
5.4 Performance on Auxiliary Tasks: Streaming ASR and Simultaneous S2TT
StreamSpeech jointly learns translation and policy through multi-task learning. As an additional prod-uct, StreamSpeech can output intermediate results of ASR and S2TT, offering users a more compre-hensive experience. To evaluate StreamSpeech’s performance on these auxiliary tasks, we present the results on the streaming ASR and Simul-S2TT tasks in Table 5 and Figure 8, respectively, assessed on the CVSS-C Fr→En test set.
For streaming ASR, StreamSpeech achieves per-formance comparable to Wav2Vec2-large (Baevski et al., 2020) and Whisper-base (Radford et al., 2022) with an average lagging of 100ms. For Simul-S2TT, StreamSpeech can generate high-quality translation with an average lagging of 2000ms. Overall, StreamSpeech excels in deliver-ing high-quality Simul-S2ST while also providing intermediate results to users as additional refer-ences. It’s important to note that although inter-mediate results can be presented, StreamSpeech does not utilize them during inference, but uses hid-den states to connect each module, making Stream-Speech a direct Simul-S2ST model.
6 Related Work
Recent research often focuses on simultaneous text-to-text (Simul-T2TT) and speech-to-text (Simul-S2TT) translation.
Simul-T2TT Simul-T2TT methods fall into fixed and adaptive. For fixed methods, Ma et al. (2019) proposed wait-k policy, which waits for k tokens before alternately READ/WRITE one token (Elbayad et al., 2020; Zheng et al., 2020; Zhang and Feng, 2021a,b). For adaptive methods, monotonic attention (Arivazhagan et al., 2019; Ma et al., 2020c; Zhang and Feng, 2022c,a), alignments (Zhang and Feng, 2023b; Guo et al., 2023a), non-autoregressive architecture (Ma et al., 2023, 2024) or language models (Guo et al., 2024a,c; Zhang et al., 2023) are employed to dynamically perform Simul-T2TT. On top of these policies, some train-ing methods are proposed to enhance the perfor-mance of policy (Zhang and Feng, 2022d; Zhang et al., 2022b; Guo et al., 2022, 2023b).
Simul-S2TT Simul-S2TT methods focus on the segmentation of speech. Ma et al. (2020b) proposed fixed pre-decision to divide speech into equal-length segments. Some adaptive methods split the speech inputs into words or segments (Ren et al., 2020; Zeng et al., 2021; Chen et al., 2021; Dong et al., 2022; Zhang and Feng, 2022b; Zhang et al., 2022a; Zhang and Feng, 2023a,c), and then apply Simul-T2TT policy. Other methods apply offline model to Simul-S2TT (Papi et al., 2023; Fu et al., 2023; Dugan et al., 2023).
We introduce StreamSpeech, an “All in One” seamless model for offline and simultaneous ASR, translation and synthesis. Compared with SOTA SeamlessStreaming (based on UnitY architecture) (Communication et al., 2023b), StreamSpeech does not design any additional simultaneous policy such as EMMA, but directly jointly learns translation and policy via multi-task learning.
7 Conclusion
In this paper, we propose StreamSpeech, an “All in One” seamless model that handles streaming ASR, simultaneous translation and real-time speech synthesis via a unified model. Experiments show the superiority of StreamSpeech on offline S2ST, streaming ASR, simultaneous S2TT and simultane-ous S2ST. Moreover, intermediate products such as ASR or S2TT results can also be presented to user during translation process as a reference, offering a better low-latency communication experience.
8972
Acknowledgements
We thank all the anonymous reviewers for their insightful and valuable comments. This work was supported by a grant from the National Natural Science Foundation of China (No. 62376260).
Limitations
In this paper, we propose StreamSpeech, a di-rect Simul-S2ST model that jointly learns transla-tion and policy in a unified framework of multi-task learning. StreamSpeech can achieve high-quality speech-to-speech translation with low la-tency. However, StreamSpeech currently focuses on synthesizing target speech with a unified voice, which limits its ability to clone the source speech’s voice characteristics. Given that voice cloning can enhance the authenticity of low-latency communi-cation, we will explore integrating voice cloning capabilities into StreamSpeech as part of future work.
References Naveen Arivazhagan, Colin Cherry, Wolfgang
Macherey, Chung-cheng Chiu, Semih Yavuz, Ruom-ing Pang, Wei Li, and Colin Raffel. 2019. Monotonic Infinite Lookback Attention for Simultaneous Machine Translation. pages 1313–1323.
Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Sys-tems, volume 33, pages 12449–12460. Curran Asso-ciates, Inc.
Junkun Chen, Mingbo Ma, Renjie Zheng, and Liang Huang. 2021. Direct simultaneous speech-to-text translation assisted by synchronized streaming ASR. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4618–4624, Online. Association for Computational Linguistics.
Kyunghyun Cho and Masha Esipova. 2016. Can neural machine translation do simultaneous translation?
Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guil-laume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi In-aguma, Somya Jain, Elahe Kalbassi, Amanda Kallet,
Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Rus-lan Mavlyutov, Benjamin Peloquin, Mohamed Ra-madan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. 2023a. Seamlessm4t: Massively multilingual & multimodal machine trans-lation.
Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoff-man, Min-Jae Hwang, Hirofumi Inaguma, Christo-pher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoari-son, Kaushik Ram Sadagopan, Abinesh Ramakr-ishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia Gonza-lez, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-jussà, Maha Elbayad, Hongyu Gong, Francisco Guzmán, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Pelo-quin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, and Mary Williamson. 2023b. Seamless: Multilin-gual expressive and streaming speech translation.
Qian Dong, Yaoming Zhu, Mingxuan Wang, and Lei Li. 2022. Learning when to translate for streaming speech. In Proceedings of the 60th Annual Meet-ing of the Association for Computational Linguistics (Volume 1: Long Papers), pages 680–694, Dublin, Ireland. Association for Computational Linguistics.
Liam Dugan, Anshul Wadhawan, Kyle Spence, Chris Callison-Burch, Morgan McGuire, and Victor Zor-dan. 2023. Learning when to speak: Latency and quality trade-offs for simultaneous speech-to-speech translation with offline models.
Maha Elbayad, Laurent Besacier, and Jakob Verbeek. 2020. Efficient Wait-k Models for Simultaneous Ma-chine Translation.
Qingkai Fang, Yan Zhou, and Yang Feng. 2023. Daspeech: Directed acyclic transformer for fast and high-quality speech-to-speech translation.
Biao Fu, Minpeng Liao, Kai Fan, Zhongqiang Huang, Boxing Chen, Yidong Chen, and Xiaodong Shi. 2023. Adapting offline speech translation models for streaming with future-aware distillation and in-ference. In Proceedings of the 2023 Conference on
8973
Empirical Methods in Natural Language Process-ing, pages 16600–16619, Singapore. Association for Computational Linguistics.
Christian Fügen, Alex Waibel, and Muntsin Kolss. 2007. Simultaneous translation of lectures and speeches. Machine translation, 21:209–252.
Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learn-ing, ICML ’06, page 369–376, New York, NY, USA. Association for Computing Machinery.
Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher. 2018. Non-autoregressive neural machine translation. In International Confer-ence on Learning Representations.
Jiatao Gu, Graham Neubig, Kyunghyun Cho, and Vic-tor O.K. Li. 2017. Learning to translate in real-time with neural machine translation. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 1053–1062, Valencia, Spain. Association for Computational Linguistics.
Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. 2020. Conformer: Convolution-augmented Trans-former for Speech Recognition. In Proc. Interspeech 2020, pages 5036–5040.
Shoutao Guo, Shaolei Zhang, and Yang Feng. 2022. Turning fixed to adaptive: Integrating post-evaluation into simultaneous machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2264–2278, Abu Dhabi, United Arab Emirates. Association for Computational Lin-guistics.
Shoutao Guo, Shaolei Zhang, and Yang Feng. 2023a. Learning optimal policy for simultaneous machine translation via binary search. In Proceedings of the 61st Annual Meeting of the Association for Compu-tational Linguistics (Volume 1: Long Papers), pages 2318–2333, Toronto, Canada. Association for Com-putational Linguistics.
Shoutao Guo, Shaolei Zhang, and Yang Feng. 2023b. Simultaneous machine translation with tailored ref-erence. In Findings of the Association for Computa-tional Linguistics: EMNLP 2023, pages 3070–3084, Singapore. Association for Computational Linguis-tics.
Shoutao Guo, Shaolei Zhang, and Yang Feng. 2024a. Decoder-only streaming transformer for simultane-ous translation. In Proceedings of the 62th Annual Meeting of the Association for Computational Lin-guistics (Long Papers), Bangkok, Thailand. Associa-tion for Computational Linguistics.
Shoutao Guo, Shaolei Zhang, and Yang Feng. 2024b. Glancing future for simultaneous machine translation. In ICASSP 2024 - 2024 IEEE International Confer-ence on Acoustics, Speech and Signal Processing (ICASSP), pages 11386–11390.
Shoutao Guo, Shaolei Zhang, Zhengrui Ma, Min Zhang, and Yang Feng. 2024c. Sillm: Large language mod-els for simultaneous machine translation.
Hirofumi Inaguma, Sravya Popuri, Ilia Kulikov, Peng-Jen Chen, Changhan Wang, Yu-An Chung, Yun Tang, Ann Lee, Shinji Watanabe, and Juan Pino. 2023. UnitY: Two-pass direct speech-to-speech translation with discrete units. In Proceedings of the 61st An-nual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15655– 15680, Toronto, Canada. Association for Computa-tional Linguistics.
Ye Jia, Michelle Tadmor Ramanovich, Tal Remez, and Roi Pomerantz. 2022a. Translatotron 2: High-quality direct speech-to-speech translation with voice preser-vation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 10120–10134. PMLR.
Ye Jia, Michelle Tadmor Ramanovich, Quan Wang, and Heiga Zen. 2022b. CVSS corpus and massively multilingual speech-to-speech translation. In Pro-ceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6691–6703, Marseille, France. European Language Resources Association.
Ye Jia, Ron J. Weiss, Fadi Biadsy, Wolfgang Macherey, Melvin Johnson, Zhifeng Chen, and Yonghui Wu. 2019. Direct Speech-to-Speech Translation with a Sequence-to-Sequence Model. In Proc. Interspeech 2019, pages 1123–1127.
Yasumasa Kano, Katsuhito Sudoh, and Satoshi Naka-mura. 2023. Average token delay: A latency metric for simultaneous translation.
Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. 2020. Hifi-gan: Generative adversarial networks for ef-ficient and high fidelity speech synthesis. In Ad-vances in Neural Information Processing Systems, volume 33, pages 17022–17033. Curran Associates, Inc.
Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tok-enizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.
Ann Lee, Peng-Jen Chen, Changhan Wang, Jiatao Gu, Sravya Popuri, Xutai Ma, Adam Polyak, Yossi Adi, Qing He, Yun Tang, Juan Pino, and Wei-Ning Hsu. 2022. Direct speech-to-speech translation with dis-crete units. In Proceedings of the 60th Annual Meet-ing of the Association for Computational Linguistics
8974
(Volume 1: Long Papers), pages 3327–3339, Dublin, Ireland. Association for Computational Linguistics.
Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. 2019. STACL: Simultaneous trans-lation with implicit anticipation and controllable la-tency using prefix-to-prefix framework. In Proceed-ings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3025–3036, Flo-rence, Italy. Association for Computational Linguis-tics.
Xutai Ma, Mohammad Javad Dousti, Changhan Wang, Jiatao Gu, and Juan Pino. 2020a. SIMULEVAL: An evaluation toolkit for simultaneous translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 144–150, Online. Association for Computational Linguistics.
Xutai Ma, Juan Pino, and Philipp Koehn. 2020b. SimulMT to SimulST: Adapting simultaneous text translation to end-to-end simultaneous speech trans-lation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Compu-tational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 582–587, Suzhou, China. Association for Computa-tional Linguistics.
Xutai Ma, Juan Miguel Pino, James Cross, Liezl Pu-zon, and Jiatao Gu. 2020c. Monotonic multihead attention. In International Conference on Learning Representations.
Zhengrui Ma, Qingkai Fang, Shaolei Zhang, Shoutao Guo, Yang Feng, and Min Zhang. 2024. A non-autoregressive generation framework for end-to-end simultaneous speech-to-any translation. In Proceed-ings of the 62th Annual Meeting of the Association for Computational Linguistics (Long Papers), Bangkok, Thailand. Association for Computational Linguistics.
Zhengrui Ma, Shaolei Zhang, Shoutao Guo, Chenze Shao, Min Zhang, and Yang Feng. 2023. Non-autoregressive streaming transformer for simultane-ous translation. In Proceedings of the 2023 Con-ference on Empirical Methods in Natural Language Processing, pages 5177–5190, Singapore. Associa-tion for Computational Linguistics.
OpenAI. 2024. Hello gpt-4o.
Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of the 2019 Con-ference of the North American Chapter of the Associa-tion for Computational Linguistics (Demonstrations), pages 48–53, Minneapolis, Minnesota. Association for Computational Linguistics.
Sara Papi, Marco Gaido, Matteo Negri, and Marco Turchi. 2022. Over-generation cannot be rewarded:
Length-adaptive average lagging for simultaneous speech translation. In Proceedings of the Third Work-shop on Automatic Simultaneous Translation, pages 12–17, Online. Association for Computational Lin-guistics.
Sara Papi, Matteo Negri, and Marco Turchi. 2023. At-tention as a guide for simultaneous speech translation. In Proceedings of the 61st Annual Meeting of the As-sociation for Computational Linguistics (Volume 1: Long Papers), pages 13340–13356, Toronto, Canada. Association for Computational Linguistics.
Sravya Popuri, Peng-Jen Chen, Changhan Wang, Juan Pino, Yossi Adi, Jiatao Gu, Wei-Ning Hsu, and Ann Lee. 2022. Enhanced Direct Speech-to-Speech Trans-lation Using Self-supervised Pre-training and Data Augmentation. In Proc. Interspeech 2022, pages 5195–5199.
Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computa-tional Linguistics.
Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukas Burget, Ondrej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlicek, Yanmin Qian, Petr Schwarz, Jan Silovsky, Georg Stemmer, and Karel Vesely. 2011. The kaldi speech recognition toolkit. IEEE Signal Processing Society. IEEE Catalog No.: CFP11SRW-USB.
Alec Radford, Jong Wook Kim, Tao Xu, Greg Brock-man, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak su-pervision.
Yi Ren, Jinglin Liu, Xu Tan, Chen Zhang, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2020. SimulSpeech: End-to-end simultaneous speech to text translation. In Proceedings of the 58th Annual Meeting of the As-sociation for Computational Linguistics, pages 3787– 3796, Online. Association for Computational Lin-guistics.
Chitwan Saharia, William Chan, Saurabh Saxena, and Mohammad Norouzi. 2020. Non-autoregressive ma-chine translation with latent alignments. In Proceed-ings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1098–1108, Online. Association for Computational Linguistics.
Elizabeth Salesky, Marcello Federico, and Marine Carpuat, editors. 2023. Proceedings of the 20th In-ternational Conference on Spoken Language Trans-lation (IWSLT 2023). Association for Computational Linguistics, Toronto, Canada (in-person and online).
Yun Tang, Juan Pino, Xian Li, Changhan Wang, and Dmitriy Genzel. 2021a. Improving speech transla-tion by understanding and learning from the auxiliary text translation task. In Proceedings of the 59th An-nual Meeting of the Association for Computational
8975
Linguistics and the 11th International Joint Confer-ence on Natural Language Processing (Volume 1: Long Papers), pages 4252–4261, Online. Association for Computational Linguistics.
Yun Tang, Juan Pino, Changhan Wang, Xutai Ma, and Dmitriy Genzel. 2021b. A general multi-task learn-ing framework to leverage text data for speech to text tasks. In ICASSP 2021 - 2021 IEEE Interna-tional Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6209–6213.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Gar-nett, editors, Advances in Neural Information Pro-cessing Systems 30, pages 5998–6008. Curran Asso-ciates, Inc.
Changhan Wang, Anne Wu, and Juan Pino. 2020. Cov-ost 2: A massively multilingual speech-to-text trans-lation corpus.
Xingshan Zeng, Liangyou Li, and Qun Liu. 2021. Real-TranS: End-to-end simultaneous speech translation with convolutional weighted-shrinking transformer. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2461–2474, Online. Association for Computational Linguistics.
Ruiqing Zhang, Zhongjun He, Hua Wu, and Haifeng Wang. 2022a. Learning adaptive segmentation policy for end-to-end simultaneous translation. In Proceed-ings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Pa-pers), pages 7862–7874, Dublin, Ireland. Association for Computational Linguistics.
Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhen-grui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. 2023. Bayling: Bridging cross-lingual align-ment and instruction following through interactive translation for large language models.
Shaolei Zhang and Yang Feng. 2021a. ICT’s system for AutoSimTrans 2021: Robust char-level simultaneous translation. In Proceedings of the Second Workshop on Automatic Simultaneous Translation, pages 1–11, Online. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2021b. Universal simul-taneous machine translation with mixture-of-experts wait-k policy. In Proceedings of the 2021 Confer-ence on Empirical Methods in Natural Language Pro-cessing, pages 7306–7317, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2022a. Gaussian multi-head attention for simultaneous machine translation. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3019–3030, Dublin, Ireland. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2022b. Information-transport-based policy for simultaneous translation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 992– 1013, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2022c. Modeling dual read/write paths for simultaneous machine transla-tion. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Vol-ume 1: Long Papers), pages 2461–2477, Dublin, Ireland. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2022d. Reducing posi-tion bias in simultaneous machine translation with length-aware framework. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6775– 6788, Dublin, Ireland. Association for Computational Linguistics.
Shaolei Zhang and Yang Feng. 2023a. End-to-end si-multaneous speech translation with differentiable seg-mentation. In Findings of the Association for Com-putational Linguistics: ACL 2023, pages 7659–7680, Toronto, Canada. Association for Computational Lin-guistics.
Shaolei Zhang and Yang Feng. 2023b. Hidden markov transformer for simultaneous machine translation. In The Eleventh International Conference on Learning Representations.
Shaolei Zhang and Yang Feng. 2023c. Unified segment-to-segment framework for simultaneous sequence generation. In Advances in Neural Information Pro-cessing Systems, volume 36, pages 45235–45258. Curran Associates, Inc.
Shaolei Zhang, Yang Feng, and Liangyou Li. 2021. Future-guided incremental transformer for simultane-ous translation. Proceedings of the AAAI Conference on Artificial Intelligence, 35(16):14428–14436.
Shaolei Zhang, Shoutao Guo, and Yang Feng. 2022b. Wait-info policy: Balancing source and target at in-formation level for simultaneous machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2249–2263, Abu Dhabi, United Arab Emirates. Association for Com-putational Linguistics.
Baigong Zheng, Kaibo Liu, Renjie Zheng, Mingbo Ma, Hairong Liu, and Liang Huang. 2020. Simultane-ous translation policies: From fixed to adaptive. In Proceedings of the 58th Annual Meeting of the Asso-ciation for Computational Linguistics, pages 2847– 2853, Online. Association for Computational Lin-guistics.
8976
A Calculation of Expected Token Number in CTC sequence
StreamSpeech leverages ASR and NAR-S2TT tasks to learn the alignment between the source speech and the source/target text, and then makes READ/WRITE decisions based on the token num-ber corresponding to the current received speech. Since the alignments are obtained through the CTC decoder, which involves blank and repeated tokens, we need to count the number of tokens that can be decoded by the CTC sequence. During inference, it is straightforward to remove duplicates and blanks from the CTC sequence to count the tokens corre-sponding to the speech. However, during training, we calculate the expected number of tokens corre-sponding to the CTC sequence.
For the excepted number of source tokens aligned to the speech inputs X , N asr
j is calculated as:
N asr j =
j∑
m=1
( 1− p (ϕ | Dasr
m )−
∑
v∈V
( p (v | Dasr
m )× p ( v | Dasr
m−1
) ) ) .
(13)
where N asr j is the number of source tokens that
align to X≤j . N asr j are calculated in an expecta-
tion manner, where p (ϕ|Dasr m ) refers to generating
blank token and ∑
v∈V(p (v|Dasr m )× p
( v|Dasr
m−1
) )
refers to generating repetitive tokens over the vo-cabulary V . Similarly, the number of target tokens that align to X≤j is calculated in the same way, denoted as N nar-s2tt
j . In particular, the probabili-ties within the CTC sequence often tend to be dis-cretized, resulting in minimal differences between the token counts in training and inference.
B Details of Ablation Study on Policy
In Sec.5.3, we conduct an ablation study on the three alignments involved in StreamSpeech, includ-ing source speech and source text, source speech and target text, target text and target speech. Here, we provide detailed explanations of the ablation study settings.
When removing the alignment between the source speech and source text, StreamSpeech is no longer wait for recognition of new tokens corre-sponding to the source speech but directly controls READ/WRITE based on the number of target to-kens corresponding to the source speech. When
StreamSpeech r = 15 r = 20 r = 25 r = 30
BLEU of units 30.48 31.08 32.00 31.55 ASR-BLEU 26.72 26.75 28.48 27.91
Table 6: Performance of various upsampling rate r on CVSS-C Fr→En validation set. “BLEU of units”: BLEU score computed on generated units sequence.
removing the alignment between the source speech and target text, StreamSpeech generates one target token once recognizing one source token. When removing the alignment between target text and target speech captured by the NAR T2U module, the second pass of StreamSpeech adopts the same autoregressive architecture as UnitY and generates units autoregressively until <eos>.
C Detailed Introduction of Baselines
Here, we give a detailed introduction to offline S2ST baselines.
S2UT (Lee et al., 2022) Speech-to-unit transla-tion (S2UT) directly translates the source speech to the target unit via one-pass architecture.
Translatotron (Jia et al., 2019) Translatotron translates the source speech to the target mel-spectrogram via one-pass architecture.
Translatotron 2 (Jia et al., 2022a) Translatotron 2 employs a two-pass architecture that generates phonemes and mel-spectrogram successively.
DASpeech (Fang et al., 2023) DASpeech em-ploys a two-pass architecture, where first performs non-autoregressive translation and then generates mel-spectrogram via fastspeech 2.
UnitY (Inaguma et al., 2023) UnitY is the state-of-the-art S2ST model, where both first and sec-ond passes apply autoregressive encoder-decoder to generate target text and units successively.
For the cascaded Simul-S2ST system, we em-ploy state-of-the-art methods, HMT from Simul-T2TT and DiSeg from Simul-S2TT, in conjunction with streaming ASR and real-time TTS module to accomplish Simul-S2ST.
ASR+HMT+TTS (Zhang and Feng, 2023b) Hidden Markov Transformer (HMT), which uses a hidden Markov model to correspond source to-kens with the target tokens, thereby learning the optimal translating moments for generating each target token.
DiSeg+TTS (Zhang and Feng, 2023a) DiSeg learns the speech segmentation from the underly-ing translation model via the differentiable segmen-
8977
Models Fr→En Es→En De→En Average greedy beam10 greedy beam10 greedy beam10 greedy beam10
ASR-BLEU
UnitY 26.90 27.77 23.93 24.95 18.19 18.74 23.01 23.82 StreamSpeech 27.58 28.45 26.16 27.25 19.72 20.93 24.49 25.54
BLASER 2.0 (Unsupervised)
UnitY 0.4467 0.4473 0.5090 0.5116 0.4431 0.4435 0.4663 0.4674 StreamSpeech 0.4486 0.4491 0.5155 0.5178 0.4514 0.4544 0.4719 0.4738
BLASER 2.0 (QE)
UnitY 3.1674 3.1772 3.3020 3.3278 3.1322 3.1537 3.2006 3.2195 StreamSpeech 3.1779 3.1872 3.3442 3.3669 3.1698 3.2033 3.2307 3.2525
BLASER 2.0 (Ref)
UnitY 3.1744 3.1965 3.2213 3.2638 2.9125 2.9372 3.1028 3.1325 StreamSpeech 3.1989 3.2200 3.3146 3.3525 3.0008 3.0482 3.1714 3.2069
Table 7: Offline S2ST performance of StreamSpeech, evaluated with BLASER 2.0.
tation, and then apply wait-k policy based on the number of speech segments.
D Upsampling Rate in NAR T2U Generation
The only hyperparameter that needs to be set in StreamSpeech is the upsampling rate r in NAR T2U generation. Table 6 reports the offline S2ST performance with different r on the CVSS-C Fr→En validation set, with r = 25 achieving the best performance. This finding is consistent with previous conclusions in non-autoregressive translation (NAT), where an upsampling rate of 2-3 times yielded the best performance (Saharia et al., 2020). Therefore, we set r = 25 in our ex-periments accordingly. The unit sequence length is approximately 10 times that of the subword se-quence length, and with a 2-3 times upsampling rate, an overall upsampling rate of around 25 times from text sequence to unit sequence is optimal.
When training StreamSpeech for a new language, it is recommended to first estimate the length ra-tio between the unit sequence and the subword sequence, and then multiply this ratio by 2-3 times to determine the appropriate upsampling rate.
E Evaluation with BLASER 2.0
Besides ASR-BLEU, we use BLASER 2.09 to as-sess the quality of the generated speech. BLASER 2.0 leverages a multilingual multimodal encoder to directly encode the speech segments for source in-put, translation output, and reference into a shared
9https://facebookresearch.github.io/stopes/ docs/eval/blaser
embedding space. It then computes a score of the translation quality that can serve as a proxy for human evaluation. BLASER 2.0 comprises three versions: Unsupervised (score 0-1), QE (score 1-5), and Ref (score 1-5). Table 7 reports the offline S2ST performance of StreamSpeech evaluated by BLASER 2.0. StreamSpeech also has significant advantages over UnitY.
F Visualization of Alignments
The policy of StreamSpeech is primarily guided by the alignment between source speech and source/target text, which is captured through the CTC decoder of the introduced ASR and NAR-S2TT tasks. We visualize the alignment captured by the CTC decoder in Figure 9.
The CTC decoder of the ASR and NAR-S2TT tasks effectively captures the alignment of speech and text and generates tokens with high accuracy, especially in terms of 1-gram accuracy. Stream-Speech starts translating upon recognizing a new source token and generates a corresponding num-ber of target words. This ensures that the received speech before translation contains complete source tokens and provides sufficient information to gen-erate target tokens.
Additionally, we observe that certain to-kens occupying the same position in the ASR and NAR-S2TT CTC sequences corre-spond to the same semantic meaning. For example, ‘début’↔‘beginning’, ‘linforma-tique’↔‘computing’ in Fr→En, ‘amante’↔‘lover’, ‘silencio’↔‘silent’ in Es→En, ‘Dort’↔‘there’, ‘back’↔‘stream’ in De→En. This suggests that
8978
(a) Case common_voice_fr_17308913 in CVSS-C Fr→En. Source transcription: laire préhistorique est le début de linformatique et est considéré comme compliquer. Target translation: the prehistoric area is the beginning of computer science and is considered to be complicated.
(b) Case common_voice_es_18307761 in CVSS-C Es→En. Source transcription: y calló tal vez esperando una disculpa amante pero yo preferí guardar silencio. Target translation: and he shut up he might have been just waiting for a loving apology but i preferred to remain silent.
(c) Case common_voice_de_17300640 in CVSS-C De→En. Source transcription: dort führt eine schmale brücke über den bach. Target translation: there a narrow bridge leads over the stream.
Figure 9: Visualization of the alignments of source speech and source/target text within the CTC decoder for ASR and NAR-S2TT tasks. Note that the positions without label refer to generating blank token ϕ, and we omit them for clarity. The vertical grey dashed lines represent chunks of 320ms.
the introduction of both source and target language CTC decoders after the encoder implicitly models cross-lingual alignments, particularly given that our introduced CTC decoder consists solely of a single fully connected layer.
G Case Study
In Figure 10, we illustrate the Simul-S2ST pro-cess of direct StreamSpeech and the cascaded “ASR+HMT+TTS” system. StreamSpeech is ca-pable of generating high-quality target speech with a delay of 2 seconds, particularly noticeable when there is prolonged silence at the beginning of the source speech. Compared to the cascaded system, direct StreamSpeech also demonstrates clear advan-
tages. Specifically, the cascaded system requires streaming ASR to first transcribe the speech into source text, then translate the current source text into target text using state-of-the-art HMT, and fi-nally synthesize the target speech. The cascad-ing of multiple modules leads to error accumula-tion, especially as the accuracy of each module in streaming scenarios generally tends to be lower than in offline scenarios. For instance, in this case, streaming ASR may incorrectly transcribe ‘cest’ as ‘ce’, leading to subsequent HMT generating the erroneous ‘fake’, ultimately resulting in incorrect speech. Therefore, for more challenging simulta-neous scenarios, direct models hold an advantage over cascaded systems.
8979
Speech Inputs:
Transcription: cest faux cest une hausse de la pression fiscale pour les familles
Reference: this is not true it is a rise in the tax burden for families _this's _wrong _this's _a _rise _of _the _pressure _fiscal _for _the _ families
cest faux cest une hausse de la pression fiscale pour les familles
ASR Results: ce
HMT Results:
faux cest une honte
de la
pression
that fake
is a
shame of
fiscale
fiscal
pour les famille
s
pressure for
families
TTS Results: that fake is a shame of fistal pressure for emilies
×
×
×
(a) Streaming ASR + HMT + Real-time TTS
Speech Inputs:
StreamSpeech:
Transcription: cest faux cest une hausse de la pression fiscale pour les familles
Reference: this is not true it is a rise in the tax burden for families _this's _wrong _this's _a _rise _of _the _pressure _fiscal _for _the _ families
cest faux cest une hausse de la pression fiscale pour les familles
that's wrong it's the progresss of fiscal pressure for families
(b) StreamSpeech
Figure 10: Case study of direct StreamSpeech and cascaded ‘ASR+HMT+TTS’. For clarity, we have marked the blue text above the source audio to represent the ground truth transcription aligned with the speech. The orange text below the target speech indicates the text transcribed by ASR-BLEU tookit.
You can hear more cases of StreamSpeech on of-fline or simultaneous speech-to-speech translation at our project page (https://ictnlp.github. io/StreamSpeech-site/).
H Configuration and Training Details
Table 8 reports the configurations of StreamSpeech and baselines. For the offline scenario, we set the chunk size of StreamSpeech to infinity and do not involve simultaneous policy, while keeping all other settings identical to the simultaneous sce-nario. For the simultaneous scenario, to evaluate Simul-S2ST under different latencies, we employ multi-chunk training to train a single StreamSpeech model and utilize this model to perform Simul-S2ST under various latency. All models are trained on 4 NVIDIA RTX 3090 GPUs.
I Latency Metrics
To more comprehensively evaluate the latency of simultaneous speech-to-speech translation, we em-ploy a variety of latency metrics, which are mainly divided into three categories: latency, computation-aware latency and streaming degree. To compute the latency metrics, we record the moment when the ith frame of the target speech is generated as ti (the starting point of the source speech consid-ered as moment 0), and X and S are the source speech and target speech, respectively. Note that all metrics are automatically calculated via SimulE-val toolkit.
Latency evaluates the duration that outputs lag behind the inputs, including:
Average Lagging (AL) (Ma et al., 2019) evalu-ates the average speech duration that target outputs
8980
lag behind the source inputs. AL is calculated as:
AL = 1
τ
τ∑
i=1
ti − i− 1
|S| / |X| ,
where τ = argmin i
(ti = |X|) . (14)
Average Proportion (AP) (Cho and Esipova, 2016) evaluates the proportion between the gen-erating moment and the total duration of source speech, calculated as:
AP = 1
|X| |S|
|S|∑
i=1
ti. (15)
Differentiable Average Lagging (DAL) (Ari-vazhagan et al., 2019) is a differentiable version of average lagging, calculated as:
DAL = 1
|S|
|S|∑
i=1
t ′ i −
i− 1
|S| / |X| ,
where t ′ i =
{ ti i = 1
max ( ti, t
′ i−1 +
|X| |S|
) i > 1
.
(16)
StartOffset measures the waiting time before outputting the first frame of target speech, calcu-lated as:
StartOffset = t1 (17)
EndOffset measures the offset of the last frame of target speech relative to the end of source speech, calculated as:
EndOffset = t|S| − |X| (18)
Length-Adaptive Average Lagging (LAAL) (Papi et al., 2022) is a modified version of the average lagging that takes into account the over-generation phenomenon, calculated as:
LAAL = 1
τ
τ∑
i=1
ti − i− 1
max(|S| , |S∗|)/ |X| ,
where τ = argmin i
(ti = |X|) , (19)
where S∗ is generated target speech. Average Token Delay (ATD) (Kano et al., 2023)
is the average delay of output sub-segments against
their corresponding input sub-segments, calculated as:
ATD = 1
|S|
|S|∑
i=1
ti − ξsegti , (20)
where ξsegti is the moment of corresponding input sub-segments of the ith output.
Computation-aware latency considers the ac-tual inference time of the model when com-puting the aforementioned latency, including: AL_CA, AP_CA, DAL_CA, StartOffset_CA, EndOffset_CA, LAAL_CA and ATD_CA.
Besides, we also evaluate the streaming degree of the generated speech. The more segments of output speech generated with shorter durations for each segment, the closer the generation is to being considered streaming. The metrics include:
Number of Chunks (NumChunks) evaluates the number of segments when generating the target speech.
Discontinuity evaluates the duration of silence produced in the generated speech while waiting for the source speech. This includes the total duration of all silences (Sum), the average duration of each silence (Ave), and the number of silence segments (Num). It’s important to note that Discontinuity is not equivalent to NumChunks. When the model finishes generating a target speech segment (i.e., a chunk), if the incoming source speech is sufficient for the model to begin translation at that moment immediately, the model will not produce disconti-nuity.
Real-time Factor (RTF) (Fügen et al., 2007) describes the ratio between the duration of outputs and inputs.
For more detailed implementations of latency computation, please refer to SimulEval toolkit10.
J Numerical Results
Tables 9, 10 and 11 report the numerical results of StreamSpeech, including more comprehensive quality and latency metrics.
ASR-BLEU (with silence) In particular, we additionally calculate the ASR-BLEU consider-ing silence for quality evaluation, denoted as ASR-BLEU (with silence). Specifically, StreamSpeech remains silent while waiting for the source speech after generating the current speech outputs. For
10https://github.com/facebookresearch/ SimulEval/blob/main/simuleval/evaluator/scorers/ latency_scorer.py
8981
instance, if StreamSpeech generates the current speech of 220ms, it will continue to wait for the streaming speech inputs of 320ms (corresponding to the chunk size). During the 100ms interval be-tween 220ms and the next 320ms chunk, Stream-Speech remains silent. ASR-BLEU (with silence) is calculated directly on these speech outputs that include silence. Note that the ASR model used in ASR-BLEU was trained on standard continuous speech, which causes a mismatch when evaluating speech with silence (may lead to some recogni-tion errors). Nevertheless, we report this metric to provide a more comprehensive evaluation of StreamSpeech.
8982
Hyperparameters S2UT Translatotron Translatotron2 DASpeech UnitY StreamSpeech
Speech Encoder
conv_kernel_sizes (5, 5) (5, 5) (5, 5) (5, 5) (5, 5) (5, 5) encoder_type Conformer Conformer Conformer Conformer Conformer Conformer encoder_layers 12 12 12 12 12 12 encoder_embed_dim 256 256 256 256 256 256 encoder_ffn_embed_dim 2048 2048 2048 2048 2048 2048 encoder_attention_heads 4 4 4 4 4 4 encoder_pos_enc_type relative relative relative relative relative relative depthwise_conv_kernel_size 31 31 31 31 31 31 streaming × × × × × ✓
Text decoder
decoder_type TransformerTransformer Transformer Transformer Transformer Transformer decoder_layers 4 4 4 4 4 4 decoder_embed_dim 512 512 512 512 512 512 decoder_ffn_embed_dim 2048 2048 2048 2048 2048 2048 decoder_attention_heads 8 8 8 8 8 8
Text-to-Speech Encoder
encoder_type - - Transformer - Transformer Transformer encoder_layers - - 2 - 2 2 encoder_embed_dim - - 512 - 512 512 encoder_ffn_embed_dim - - 2048 - 2048 2048 encoder_attention_heads - - 8 - 8 8
Acoustic Decoder
output_type unit (1000)
mel-spectrogram
mel-spectrogram
mel-spectrogram
unit (1000)
unit (1000)
decoder_layers 6 6 6 6 2 2 decoder_embed_dim 512 512 512 512 512 512 decoder_ffn_embed_dim 2048 2048 2048 2048 2048 2048 decoder_attention_heads 8 8 8 4 8 8
Training
lr 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 lr_scheduler inverse_sqrt inverse_sqrt inverse_sqrt inverse_sqrt inverse_sqrt inverse_sqrt warmup_updates 4000 4000 4000 4000 4000 4000 warmup_init_lr 1e-7 1e-7 1e-7 1e-7 1e-7 1e-7 optimizer Adam Adam Adam Adam Adam Adam dropout 0.1 0.1 0.1 0.1 0.1 0.1 weight_decay 0.0 0.0 0.0 0.0 0.0 0.0 clip_norm 1.0 1.0 1.0 1.0 1.0 1.0 max_tokens 160k 160k 160k 160k 160k 160k s2st_loss_weight 1.0 1.0 1.0 5.0 1.0 1.0 s2tt_loss_weight 8.0 0.1 0.1 1.0 8.0 8.0 nar_s2tt_loss_weight - - - - - 4.0 asr_loss_weight - - - - - 4.0
Table 8: Configuration of StreamSpeech and baselines.
8983
CVSS-C Fr→En
C×40ms ASR-BLEU Latency
AL AP DAL StartOffset EndOffset LAAL ATD
320 ms 22.89 1269.84 0.52 1702.30 1667.11 699.22 1358.15 2290.69 640 ms 24.41 2326.17 0.40 1946.50 1888.58 1030.48 2332.34 2669.34 960 ms 25.00 2803.13 0.35 2124.49 2076.37 1107.48 2806.27 2862.42 1280 ms 25.20 3146.27 0.31 2309.87 2231.73 1211.81 3148.17 3079.90 1600 ms 25.30 3287.21 0.29 2352.54 2215.25 1321.50 3288.82 3240.42 1920 ms 25.50 3450.24 0.27 2477.95 2296.93 1436.70 3451.45 3381.28 2240 ms 25.50 3629.71 0.25 2666.28 2471.04 1545.23 3630.60 3520.22 2560 ms 25.68 3812.13 0.24 2891.02 2695.36 1639.45 3812.69 3651.32 2880 ms 25.60 3992.35 0.22 3131.35 2957.32 1719.35 3992.80 3776.42 3200 ms 25.75 4157.28 0.22 3370.39 3228.57 1800.14 4157.49 3908.66 4800 ms 26.14 4873.08 0.18 4505.76 4490.42 2250.83 4873.08 4640.12 10000 ms 26.20 5683.92 0.13 5683.92 5683.92 3096.54 5683.92 5672.35
C×40ms ASR-BLEU Computation-Aware Latency
AL_CA AP_CA DAL_CA StartOffset_CA EndOffset_CA LAAL_CA ATD_CA
320 ms 22.89 2195.04 0.68 2333.37 2052.33 699.22 2269.02 2840.15 640 ms 24.41 2908.52 0.47 2369.64 2164.57 1030.48 2913.75 2958.70 960 ms 25.00 3331.47 0.41 2548.80 2363.50 1107.48 3334.27 3133.06 1280 ms 25.20 3576.95 0.35 2680.58 2469.84 1211.81 3578.67 3292.64 1600 ms 25.30 3694.33 0.33 2756.06 2451.59 1321.50 3695.72 3449.34 1920 ms 25.50 3765.89 0.29 2777.31 2465.22 1436.70 3766.86 3540.84 2240 ms 25.50 3995.11 0.28 3029.87 2677.67 1545.23 3995.87 3725.77 2560 ms 25.68 4167.48 0.26 3229.14 2920.50 1639.45 4168.00 3837.85 2880 ms 25.60 4323.54 0.24 3430.48 3170.37 1719.35 4323.93 3951.69 3200 ms 25.75 4502.11 0.23 3671.07 3406.31 1800.14 4502.32 4105.75 4800 ms 26.14 5189.20 0.19 4752.24 4723.36 2250.83 5189.20 4815.85 10000 ms 26.20 5946.97 0.13 5946.97 5946.97 3096.54 5946.97 5816.81
C×40ms ASR-BLEU Streaming Degree
Num Chunks RTF Discontinuity
Sum Discontinuity
Ave Discontinuity
Num
320 ms 22.89 7.85 1.15 1695.99 385.77 4.42 640 ms 24.41 5.43 1.20 1745.35 549.87 3.28 960 ms 25.00 4.46 1.22 1630.69 585.05 2.72 1280 ms 25.20 3.83 1.24 1583.53 672.25 2.23 1600 ms 25.30 3.48 1.26 1705.61 843.01 1.93 1920 ms 25.50 3.15 1.29 1736.51 997.20 1.64 2240 ms 25.50 2.86 1.30 1668.21 1101.78 1.37 2560 ms 25.68 2.62 1.32 1543.38 1144.47 1.16 2880 ms 25.60 2.40 1.34 1361.69 1103.53 0.97 3200 ms 25.75 2.23 1.35 1166.73 1009.29 0.80 4800 ms 26.14 1.69 1.44 356.97 354.08 0.25 10000 ms 26.20 1.00 1.56 0.00 0.00 0.00
C×40ms ASR-BLEU Quality with other Metrics
ASR-BLEU (with silence)
BLASER 2.0 (Unsupervised)
BLASER 2.0 (QE)
BLASER 2.0 (Ref)
320 ms 22.89 17.88 0.4428 3.1170 3.0519 640 ms 24.41 19.65 0.4439 3.1322 3.0965 960 ms 25.00 20.20 0.4446 3.1373 3.1067 1280 ms 25.20 21.14 0.4448 3.1391 3.1109 1600 ms 25.30 21.07 0.4450 3.1420 3.1164 1920 ms 25.50 21.61 0.4451 3.1429 3.1195 2240 ms 25.50 21.97 0.4451 3.1443 3.1226 2560 ms 25.68 22.76 0.4456 3.1464 3.1268 2880 ms 25.60 23.26 0.4456 3.1484 3.1294 3200 ms 25.75 23.91 0.4456 3.1475 3.1294 4800 ms 26.14 25.66 0.4462 3.1530 3.1397 10000 ms 26.20 26.20 0.4465 3.1569 3.1486
Table 9: Numerical results of StreamSpeech on CVSS-C Fr→En.
8984
CVSS-C Es→En
C×40ms ASR-BLEU Latency
AL AP DAL StartOffset EndOffset LAAL ATD
320 ms 20.06 1522.05 0.52 1899.15 1829.94 811.60 1611.94 2647.52 640 ms 21.68 2514.69 0.40 2129.15 2050.91 1082.63 2522.64 3000.61 960 ms 22.36 2999.86 0.35 2274.76 2207.81 1138.16 3002.95 3189.89 1280 ms 22.76 3410.10 0.31 2510.28 2438.99 1218.14 3411.50 3400.50 1600 ms 22.94 3577.51 0.29 2566.79 2433.17 1310.75 3578.60 3571.73 1920 ms 23.19 3708.04 0.27 2632.80 2442.63 1423.14 3709.03 3716.76 2240 ms 23.26 3870.11 0.25 2785.82 2564.04 1516.54 3870.45 3846.65 2560 ms 23.46 4050.40 0.23 2992.91 2766.25 1616.86 4050.55 3982.72 2880 ms 23.51 4236.50 0.22 3232.40 3019.76 1694.92 4236.58 4108.61 3200 ms 23.58 4408.96 0.21 3476.29 3289.87 1766.69 4409.03 4227.52 4800 ms 23.97 5161.83 0.18 4677.54 4654.39 2131.72 5161.83 4909.06 10000 ms 24.22 6185.38 0.12 6185.38 6185.38 3118.92 6185.38 6171.87
C×40ms ASR-BLEU Computation-Aware Latency
AL_CA AP_CA DAL_CA StartOffset_CA EndOffset_CA LAAL_CA ATD_CA
320 ms 20.06 2395.05 0.66 2474.80 2177.62 811.60 2471.77 3041.25 640 ms 21.68 3224.90 0.49 2751.52 2425.16 1082.63 3231.63 3428.89 960 ms 22.36 3462.63 0.40 2625.00 2414.63 1138.16 3465.45 3410.61 1280 ms 22.76 3826.98 0.35 2844.11 2653.91 1218.14 3828.28 3590.27 1600 ms 22.94 3974.31 0.32 2945.91 2637.41 1310.75 3975.25 3782.11 1920 ms 23.19 4048.74 0.29 2964.03 2622.29 1423.14 4049.54 3895.00 2240 ms 23.26 4211.71 0.27 3124.19 2746.27 1516.54 4212.04 4035.37 2560 ms 23.46 4348.39 0.25 3282.05 2912.61 1616.86 4348.52 4151.34 2880 ms 23.51 4556.30 0.24 3521.85 3176.94 1694.92 4556.37 4291.52 3200 ms 23.58 4725.99 0.23 3765.53 3468.13 1766.69 4726.06 4412.92 4800 ms 23.97 5487.24 0.19 4921.40 4888.64 2131.72 5487.24 5086.99 10000 ms 24.22 6472.57 0.12 6472.57 6472.57 3118.92 6472.57 6329.43
C×40ms ASR-BLEU Streaming Degree
Num Chunks RTF Discontinuity
Sum Discontinuity
Ave Discontinuity
Num
320 ms 20.06 8.02 1.15 2141.84 455.29 5.01 640 ms 21.68 5.74 1.19 2119.97 600.15 3.81 960 ms 22.36 4.75 1.20 2010.21 659.49 3.16 1280 ms 22.76 4.02 1.21 1857.09 728.14 2.52 1600 ms 22.94 3.63 1.23 1953.86 880.22 2.16 1920 ms 23.19 3.33 1.25 2050.74 1068.46 1.88 2240 ms 23.26 3.05 1.27 2024.67 1226.73 1.60 2560 ms 23.46 2.79 1.28 1921.86 1326.79 1.36 2880 ms 23.51 2.56 1.29 1751.37 1357.80 1.15 3200 ms 23.58 2.38 1.30 1546.32 1298.67 0.99 4800 ms 23.97 1.81 1.37 539.01 534.01 0.37 10000 ms 24.22 1.00 1.52 0.00 0.00 0.00
C×40ms ASR-BLEU Quality with other Metrics
ASR-BLEU (with silence)
BLASER 2.0 (Unsupervised)
BLASER 2.0 (QE)
BLASER 2.0 (Ref)
320 ms 20.06 14.76 0.5011 3.2251 3.0946 640 ms 21.68 16.57 0.5053 3.2593 3.1567 960 ms 22.36 17.49 0.5072 3.2722 3.1783 1280 ms 22.76 18.29 0.5074 3.2767 3.1856 1600 ms 22.94 18.64 0.5083 3.2812 3.1921 1920 ms 23.19 18.93 0.5085 3.2842 3.1994 2240 ms 23.26 19.29 0.5089 3.2866 3.2052 2560 ms 23.46 20.12 0.5093 3.2908 3.2099 2880 ms 23.51 20.80 0.5099 3.2943 3.2157 3200 ms 23.58 21.22 0.5102 3.2957 3.2176 4800 ms 23.97 23.29 0.5108 3.3017 3.2295 10000 ms 24.22 24.22 0.5114 3.3075 3.2397
Table 10: Numerical results of StreamSpeech on CVSS-C Es→En.
8985
CVSS-C De→En
C×40ms ASR-BLEU Latency
AL AP DAL StartOffset EndOffset LAAL ATD
320 ms 14.56 1687.62 0.46 1815.81 1758.31 1186.14 1741.47 2736.31 640 ms 15.83 2561.63 0.36 2078.04 2004.57 1327.14 2566.84 3042.11 960 ms 16.34 2978.14 0.32 2256.35 2189.65 1361.72 2980.68 3202.18 1280 ms 16.57 3276.92 0.29 2424.27 2341.71 1426.29 3279.19 3379.61 1600 ms 16.75 3418.49 0.28 2477.75 2341.77 1506.44 3420.34 3516.78 1920 ms 16.85 3568.48 0.26 2597.41 2426.33 1587.89 3569.64 3640.85 2240 ms 17.02 3736.77 0.24 2777.56 2591.56 1668.12 3737.84 3758.86 2560 ms 17.17 3904.82 0.23 2982.85 2800.69 1733.56 3905.40 3868.65 2880 ms 17.23 4060.73 0.22 3193.25 3027.93 1802.33 4061.25 3984.06 3200 ms 17.16 4219.67 0.21 3425.01 3282.92 1862.79 4220.07 4102.51 4800 ms 17.52 4916.03 0.18 4519.45 4500.36 2205.11 4916.18 4736.56 10000 ms 18.05 5741.25 0.12 5741.25 5741.25 2968.60 5741.25 5730.22
C×40ms ASR-BLEU Computation-Aware Latency
AL_CA AP_CA DAL_CA StartOffset_CA EndOffset_CA LAAL_CA ATD_CA
320 ms 14.56 2558.81 0.59 2446.13 2135.38 1186.14 2604.99 3196.74 640 ms 15.83 3161.91 0.43 2524.26 2284.92 1327.14 3166.60 3338.35 960 ms 16.34 3448.73 0.36 2590.87 2418.56 1361.72 3450.96 3407.84 1280 ms 16.57 3677.11 0.33 2735.80 2552.65 1426.29 3679.01 3561.59 1600 ms 16.75 3752.73 0.30 2764.69 2510.66 1506.44 3754.44 3681.01 1920 ms 16.85 3883.72 0.28 2901.05 2597.63 1587.89 3884.82 3805.99 2240 ms 17.02 4009.48 0.26 3024.46 2736.61 1668.12 4010.50 3905.44 2560 ms 17.17 4174.97 0.25 3221.65 2952.58 1733.56 4175.52 4014.60 2880 ms 17.23 4339.98 0.24 3444.19 3203.36 1802.33 4340.44 4147.01 3200 ms 17.16 4484.03 0.23 3654.72 3448.12 1862.79 4484.39 4253.07 4800 ms 17.52 5234.16 0.19 4766.98 4741.92 2205.11 5234.31 4912.98 10000 ms 18.05 5977.26 0.13 5977.26 5977.26 2968.60 5977.26 5860.21
C×40ms ASR-BLEU Streaming Degree
Num Chunks RTF Discontinuity
Sum Discontinuity
Ave Discontinuity
Num
320 ms 14.56 6.85 1.24 2246.02 565.17 4.39 640 ms 15.83 4.93 1.26 2092.95 703.44 3.24 960 ms 16.34 4.15 1.27 1942.50 734.28 2.71 1280 ms 16.57 3.64 1.28 1851.17 800.32 2.25 1600 ms 16.75 3.34 1.30 1926.99 948.35 1.96 1920 ms 16.85 3.06 1.31 1922.77 1074.24 1.69 2240 ms 17.02 2.81 1.33 1832.01 1165.29 1.43 2560 ms 17.17 2.58 1.34 1695.21 1200.79 1.22 2880 ms 17.23 2.39 1.35 1531.03 1182.17 1.04 3200 ms 17.16 2.23 1.36 1339.35 1112.27 0.88 4800 ms 17.52 1.68 1.43 482.26 478.56 0.32 10000 ms 18.05 1.00 1.54 0.00 0.00 0.00
C×40ms ASR-BLEU Quality with other Metrics
ASR-BLEU (with silence)
BLASER 2.0 (Unsupervised)
BLASER 2.0 (QE)
BLASER 2.0 (Ref)
320 ms 14.56 10.81 0.4393 3.0864 2.8424 640 ms 15.83 12.14 0.4419 3.1041 2.8808 960 ms 16.34 12.79 0.4435 3.1141 2.8993 1280 ms 16.57 13.35 0.4436 3.1166 2.9045 1600 ms 16.75 13.49 0.4441 3.1203 2.9109 1920 ms 16.85 13.87 0.4445 3.1232 2.9148 2240 ms 17.02 14.34 0.4448 3.1234 2.9200 2560 ms 17.17 14.84 0.4447 3.1234 2.9217 2880 ms 17.23 15.31 0.4450 3.1267 2.9254 3200 ms 17.16 15.62 0.4450 3.1253 2.9248 4800 ms 17.52 17.09 0.4459 3.1311 2.9361 10000 ms 18.05 18.05 0.4469 3.1394 2.9507
Table 11: Numerical results of StreamSpeech on CVSS-C De→En.
8986