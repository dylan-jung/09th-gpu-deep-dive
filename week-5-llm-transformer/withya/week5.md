# Week 5 복습: LLM 추론 구조와 병목 이해

---

## 1. 학습 vs 추론: 왜 먼저 구분하는가?

GPU를 많이 쓴다는 건 같은데 반복 구조가 완전히 다르다.

**학습**은 "데이터 → forward → loss → backward → gradient 동기화 → weight 업데이트"를 반복하는 거다. 여기서 핵심 병목은 rank 간 gradient AllReduce인데, 모든 GPU가 동기화될 때까지 기다려야 하기 때문이다.

**추론**은 "요청이 들어오면 첫 token을 빨리 내고, 이후 token을 일정 간격으로 계속 내는" 반복이다. gradient가 없고 대신 **KV cache**가 중심 자원이 된다.

| 구분 | 학습 | 추론 |
|---|---|---|
| 반복 단위 | training step | request → prefill → decode |
| 주요 상태 | weight, gradient, optimizer state | weight, KV cache, request queue |
| 대표 통신 | gradient AllReduce | TP collective, KV transfer |
| 주 지표 | step time, samples/sec | TTFT, TPOT/ITL, p95/p99 |

Week 4에서 배운 NCCL/collective가 학습에서는 gradient 동기화로 쓰이고, 추론에서는 tensor parallel collective나 KV transfer로 다시 등장한다.

---

## 2. LLM 추론 흐름 전체 그림

```
사용자 입력 → tokenize → Prefill → Decode 반복 → 응답 스트리밍
```

Prefill이랑 Decode가 같은 transformer를 통과하는데, **병목 성격이 완전히 다르다**. 이게 이번 강의 전체의 핵심이다.

tokenize 단계에서도 병목이 날 수 있다. tokenizer thread가 밀리거나 queue가 쌓이면 GPU는 멀쩡한데 첫 token이 늦어지는 상황이 생긴다.

---

## 3. Transformer layer 안에서 무슨 일이 일어나는가?

각 layer는 대략 이 순서로 흐른다.

```
hidden state
→ Norm (값 안정화)
→ Q/K/V projection (attention용 재료 만들기)
→ Attention (지금 token이 과거 중 뭘 볼지 결정)
→ Output projection
→ Residual add
→ Norm
→ MLP / FFN (token별 비선형 변환)
→ Residual add
→ (마지막 layer라면) LM head → logits → sampling → 다음 token
```

**Logits와 Sampling을 헷갈리지 말 것.** logits는 vocab별 raw score고, 거기에 temperature/top-k/top-p를 적용한 다음 softmax로 확률 분포를 만들고, 그 분포에서 실제 token을 하나 고르는 게 sampling이다.

Attention에서 Q, K, V가 뭔지도 정리하면:
- Q = 지금 token이 던지는 질문
- K = 과거 token들의 색인
- V = 과거 token들의 내용

Q와 K를 비교해서 어떤 과거 token을 많이 볼지 정하고, 그 가중치로 V를 모아서 다음 hidden state를 만든다.

---

## 4. Prefill: compute-bound

Prefill은 prompt 전체를 **한 번에** 처리한다. prompt가 8,000 token이면 8,000개를 동시에 transformer에 넣고 각 layer의 K/V를 생성한다.

token이 많으니까 행렬 곱(GEMM)이 커지고, Tensor Core를 꽉 채울 수 있다. 그래서 **compute-bound**다. 병목은 Tensor Core 계산 자체와 attention IO다.

느리면 TTFT가 나빠진다. 해결책은 FlashAttention (attention kernel 최적화), chunked prefill, FP8/BF16 kernel, prefix caching (반복되는 system prompt 재사용) 이런 것들이다.

---

## 5. Decode: HBM bandwidth-bound

Decode는 output token을 **하나씩** 만든다. 다음 token은 이전 token에 의존하므로 완전한 병렬화가 불가능하다.

```
step 1: token_1 생성
step 2: token_1 보고 token_2 생성
step 3: token_1, token_2 보고 token_3 생성
...
```

문제는 매 step마다 **모든 weight + 지금까지 쌓인 KV cache 전체**를 HBM에서 읽어야 하는데, 정작 한 번에 계산하는 건 token 1개뿐이라는 거다. Tensor Core 입장에서는 엄청 비효율적이다. 먹을 게 1개뿐인데 밥상을 매번 처음부터 다 차려야 하는 상황.

그래서 계산이 느린 게 아니라 **HBM에서 데이터를 읽어오는 속도**가 병목이 된다. 이게 HBM bandwidth-bound다.

A100(~2.0 TB/s)보다 H100(~3.35 TB/s)에서 decode가 더 빠른 이유가 바로 이거다. 같은 80GB HBM이라도 bandwidth가 다르다.

느리면 TPOT/ITL이 나빠진다. 해결책은 continuous batching, GQA/MQA, KV quantization, speculative decoding 등이다.

### Roofline으로 정리

```
Decode  ← 낮은 arithmetic intensity → HBM bandwidth-bound
Prefill ← 높은 arithmetic intensity → Tensor Core compute-bound
```

Prefill이 느리면 계산과 attention kernel을 보고, Decode가 느리면 HBM bandwidth, KV cache, batching을 먼저 봐야 한다. 같은 방식으로 튜닝하면 안 된다.

---

## 6. KV Cache: 절약이지만 공짜가 아니다

KV cache가 없으면 decode step마다 prompt 전체를 다시 계산해야 한다. 그러면 너무 비효율적이니까 이미 읽은 token의 K/V activation을 저장해두는 거다.

compute는 절약되지만, 대신 **HBM capacity와 HBM bandwidth를 계속 잡아먹는다**. 운영할 때 "모델 weight가 GPU에 들어가는가?"만 보면 부족한 이유가 여기 있다. weight + KV cache + activation workspace + fragmentation 여유가 모두 HBM에 들어가야 한다.

### KV cache 크기 공식

```
KV bytes = layers × batch × seq_len × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
```

**선형 관계가 핵심이다:**
- context length 4배 → KV 4배
- batch 2배 → KV 2배
- num_kv_heads 절반 → KV 절반

예를 들어 70B GQA 모델에서 128k context, batch=1이면 KV만 약 40 GiB다. H100 80GB에 weight 약 140 GiB(BF16 기준)가 이미 올라가 있으면 KV가 들어갈 공간 자체가 없다. 이 때문에 보통 FP8이나 quantization으로 weight를 줄이거나, KV quantization을 쓴다.

### MHA vs GQA vs MQA

기본 MHA는 query head 수만큼 KV head도 있어서 KV가 크다. GQA는 여러 query head가 더 적은 KV head를 공유한다. MQA는 극단적으로 KV head를 거의 1개로 줄인다.

요즘 LLM (Llama 3, Mistral 등)이 GQA를 쓰는 이유가 여기 있다. KV 저장량과 KV read bytes가 줄어서 decode HBM bandwidth 부담이 낮아진다.

### KV cache 최적화 기법들

- **PagedAttention**: 요청 길이가 달라서 생기는 메모리 fragmentation을 줄인다. 고정 크기 page 단위로 KV를 관리.
- **Prefix caching**: 반복되는 system prompt나 RAG 템플릿의 KV를 재사용한다. prefill compute를 아끼고 TTFT도 좋아진다.
- **KV quantization**: KV를 INT8/FP8로 저장해서 용량과 bandwidth를 줄인다.
- **CPU DRAM offload**: HBM이 부족할 때 cold KV를 host DRAM으로 내린다. PCIe를 타야 해서 p99가 커질 수 있다.

---

## 7. 성능 지표: 하나의 숫자로 설명하면 안 된다

| 지표 | 의미 | 주로 보는 병목 |
|---|---|---|
| TTFT | 요청 후 첫 token까지 걸리는 시간 | queue, prefill, first decode |
| TPOT / ITL | 첫 token 이후 token 간 간격 | decode loop, HBM bandwidth |
| 전체 Latency | TTFT + output_length × TPOT | 둘 다 |
| p99 | 99% 요청이 이 시간 안에 끝나는 tail latency | congestion, 긴 prompt, straggler |
| Token throughput | 초당 처리 token 수 | GPU utilization, batch size |

### batch size trade-off

batch가 커지면 GPU utilization이 올라가고 token throughput도 오른다. 그런데 동시에 queue 대기 시간이 늘고 KV footprint도 커져서 TTFT와 p99가 나빠질 수 있다. 목적에 따라 조절해야 한다.

- Chatbot/코딩 어시스턴트: TTFT, ITL, p99 우선
- RAG/Document QA: TTFT, goodput 우선
- Offline batch: TPS, cost/token 우선

---

## 8. Batching 전략

### Static batching

한 묶음이 끝날 때까지 기다리는 방식이다.

```
Req A: [work work work work]  ← 긴 요청
Req B: [work work PAD  PAD ]  ← 짧은 요청이 A 끝날 때까지 기다림
Req C: [work WAIT WAIT WAIT]
```

짧은 요청이 긴 요청에 묶여 대기하고, padding 때문에 GPU slot도 낭비된다.

### Continuous batching

decode iteration마다 완료된 요청을 빼고 새 요청을 넣는다.

```
t1: A decode | B decode | C prefill
t2: A decode | C decode | D prefill  ← B 완료 즉시 D 투입
t3: C decode | D decode | E prefill
```

join/leave 단위가 request 전체가 아니라 **decode iteration**이다. GPU utilization을 높이면서 decode cadence를 유지하는 전략이다. 하지만 무작정 batch를 키우면 KV cache와 queue도 같이 커진다.

### Chunked prefill

긴 prompt가 들어오면 prefill이 decode를 오래 막는다. 이걸 여러 chunk로 잘라서 decode slot 사이에 끼워 넣는다.

```
Long prompt = P1 + P2 + P3

iteration 1: [P1 prefill] [A decode] [B decode]
iteration 2: [P2 prefill] [A decode] [B decode]
iteration 3: [P3 prefill] [A decode] [B decode]
```

TTFT와 TPOT/ITL 사이에서 균형을 잡는 전략이다. prefill이 decode를 독점하면 ITL/p99가 나빠지기 때문.

---

## 9. Multi-GPU: TP / PP / DP

여러 GPU를 쓸 때 "모델을 어떤 축으로 나누는가"로 이해하면 된다.

### TP (Tensor Parallelism): layer 내부 tensor 축

각 layer 안의 큰 weight/activation 차원을 여러 GPU가 나눠 계산한다.

```
Layer 1:  GPU0=W의 절반 | GPU1=W의 나머지 절반
Layer 2:  GPU0=W의 절반 | GPU1=W의 나머지 절반
...
→ 각 layer마다 partial result를 all-reduce/all-gather로 합친다
```

layer마다 동기화가 들어가기 때문에 **NVLink/NVSwitch bandwidth가 병목**이다. 그래서 TP는 가능하면 같은 NVLink/NVSwitch domain 안에 묶어야 한다. node 밖으로 나가면 IB/RDMA latency가 그대로 decode latency에 들어온다.

### PP (Pipeline Parallelism): layer 깊이 축

전체 layer stack을 stage 단위로 나눈다.

```
GPU0: Layer 1-10
GPU1: Layer 11-20
GPU2: Layer 21-30
GPU3: Layer 31-40
```

각 stage가 자기 layer를 처리하고 activation을 다음 stage로 넘긴다. microbatch가 충분하지 않거나 stage 시간이 불균형하면 **pipeline bubble**이 생겨 GPU가 쉰다. 모델이 한 노드에 안 들어갈 때 노드 경계에 맞춰 쓴다.

### DP (Data Parallelism): request/replica 축

모델 전체 replica를 여러 GPU/노드에 복제하고 request를 나눠 보낸다.

```
Replica 0: full model → Req A
Replica 1: full model → Req B
Replica 2: full model → Req C
```

추론에서는 학습처럼 gradient all-reduce가 핵심이 아니다. 대신 replica마다 weight와 KV cache를 들고 있어야 해서 **HBM capacity가 replica 수에 비례해 필요하다**. 모델이 replica 단위로 한 장에 들어가고 traffic이 많으면 가장 단순하고 강력한 방법이다.

### 상황별 선택

| 상황 | 선택 |
|---|---|
| 7B/8B, GPU 한 장에 들어감 | single GPU 또는 DP |
| 70B, 한 GPU에 안 들어감 | node 내부 TP=2/4 |
| 405B+, 노드 하나도 어려움 | TP+PP, quantization, multi-node |
| long context KV가 너무 큼 | GQA/MQA, KV quant, offload 검토 |

### CPU/NVMe offload가 자주 느린 이유

HBM → PCIe → CPU DRAM → NVMe 순으로 갈수록 크지만 느려진다. hot KV나 매 layer weight가 이 경로를 반복해서 지나면 PCIe/NVMe가 바로 병목이 된다. offload는 hot path가 아니라 cold KV, reusable prefix, capacity overflow를 다루는 보조 수단으로 쓰는 게 안전하다.

---

## 10. PD Disaggregation

### Colocated serving의 문제

Prefill과 Decode를 같은 GPU pool에서 같이 돌리면 자원 경쟁이 생긴다.
- Prefill은 compute-heavy
- Decode는 bandwidth-heavy이고 latency-sensitive

긴 prefill batch가 들어오면 decode가 밀려서 ITL/p99가 흔들린다.

### Prefill pool / Decode pool 분리

```
Router → Prefill GPU pool → KV transfer (RDMA/NVLink) → Decode GPU pool → 응답
```

분리하면 prefill/decode 자원 경쟁이 없어지고, 각 pool을 목적에 맞게 sizing할 수 있다. SLO별 routing과 cache locality도 활용할 수 있다.

### 새로 생기는 문제: KV transfer

prefill이 만든 KV를 decode 쪽으로 보내야 한다. 이게 새로운 병목이 된다.

70B GQA, BF16, batch=1 기준:

| Context | KV 크기 | 100G | 200G | 400G |
|---|---|---|---|---|
| 8k | ~2.5 GiB | ~268 ms | ~134 ms | ~67 ms |
| 32k | ~10 GiB | ~1.07 s | ~537 ms | ~268 ms |
| 128k | ~40 GiB | ~4.3 s | ~2.15 s | ~1.07 s |

8k도 interactive TTFT에 영향이 보일 수 있고, 32k 이상에서는 KV compression이나 prefix locality 없이는 PD 자체가 TTFT 병목이 될 수 있다. 128k는 400G 네트워크도 1초가 넘는다.

---

## 11. 운영 디버깅 순서

### TTFT가 나쁘다

1. queue time → 2. tokenization time → 3. prompt length 분포 → 4. prefill latency → 5. prefix cache hit ratio → 6. (PD라면) KV transfer time

레버: chunked prefill, prefix caching, admission control, prefill pool scale-out

### TPOT/ITL이 나쁘다

1. decode batch size → 2. HBM bandwidth utilization → 3. KV cache bytes/request → 4. sampling/logits overhead → 5. TP collective latency

레버: continuous batching, KV quantization, GQA/MQA 모델, speculative decoding, HBM bandwidth가 더 큰 GPU

### OOM 또는 max concurrency가 낮다

1. model weight memory → 2. KV cache memory → 3. context length → 4. batch/concurrency → 5. fragmentation → 6. CUDA graph/workspace 여유

레버: max context 제한, PagedAttention, KV quantization, prefix eviction, cold KV offload

### Multi-node scaling이 안 된다

1. TP traffic이 node 밖으로 나가는가? → 2. NCCL이 NVLink/IB/GDRDMA path를 쓰는가? → 3. TCP/socket fallback이 있는가? → 4. GPU-NIC locality가 맞는가? → 5. RoCE라면 PFC/ECN/MTU/GID 설정이 맞는가?

레버: TP를 NVLink domain 내부로 제한, PP/DP로 분할 방식 변경, GPUDirect RDMA 확인

---

## 12. 전체 핵심 정리

1. LLM 추론은 **Prefill**과 **Decode**로 나눠 봐야 한다.
2. Prefill은 prompt 전체를 병렬 처리 → **Tensor Core compute-bound** → TTFT에 영향
3. Decode는 token 하나씩 + weight/KV 반복 읽기 → **HBM bandwidth-bound** → TPOT에 영향
4. KV cache는 compute를 절약하지만 HBM capacity와 bandwidth를 계속 소비한다.
5. context length, batch, concurrency는 KV cache를 **선형으로** 키운다.
6. GQA/MQA는 KV head 수를 줄여 KV 저장량과 read bytes를 낮춘다. decode에서 특히 유리하다.
7. Continuous batching은 GPU utilization을 높이지만 KV footprint와 queue를 함께 봐야 한다.
8. TP = layer 내부 tensor 축 (NVLink 병목), PP = layer 깊이 축 (pipeline bubble 병목), DP = request/replica 축 (HBM capacity 병목).
9. TP는 반드시 같은 NVLink/NVSwitch domain 안에 묶어야 한다. 밖으로 나가면 IB latency가 decode latency에 직접 들어온다.
10. PD disaggregation은 prefill/decode 자원 경쟁을 줄이지만 **KV transfer**라는 새 병목을 만든다. context가 길수록 치명적이다.

---

*원본 강의 노트: `w5-llm-inference-bottlenecks-lecture.md`*
