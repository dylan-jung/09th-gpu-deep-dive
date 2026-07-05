# Week 5 LLM 추론 구조와 병목

---

## 1. Prefill과 Decode의 병목 자원이 왜 다른가


같은 transformer를 지나지만 연산 모양(arithmetic intensity)이 다르기 때문이다.

| 단계 | 처리 방식 | 포화되는 자원 | 이유 |
|---|---|---|---|
| Prefill | prompt 전체 token을 한 번에 처리 | **Tensor Core compute** | 많은 token을 한 번에 큰 batched GEMM으로 계산 → 계산기를 꽉 채움 (compute-bound) |
| Decode | output token을 1개씩 생성 | **HBM bandwidth / capacity** | 매 step마다 weight와 과거 KV를 반복해서 읽지만, 한 번에 계산할 token이 적어 메모리 읽기 속도가 앞을 막음 (bandwidth-bound) |

핵심 직관(roofline):
- Prefill = **계산을 많이 함** → 높은 arithmetic intensity → compute-bound
- Decode = **bytes를 많이 읽음** → 낮은 arithmetic intensity → HBM bandwidth-bound

따라서 둘은 **같은 방식으로 튜닝하면 안 된다.** Prefill이 느리면 계산·attention kernel을, Decode가 느리면 HBM bandwidth·KV·batching을 먼저 본다.

---

## 2. TTFT와 TPOT/ITL을 구분해 병목을 추론하기

두 지표는 추론 루프의 **다른 구간**을 본다.

| 지표 | 의미 | 보는 구간 | 주 병목 |
|---|---|---|---|
| **TTFT** (Time To First Token) | 요청 후 첫 token까지 | queue → tokenize → prefill → first decode | queue, tokenization, prefill compute, (PD라면) KV transfer |
| **TPOT / ITL** (Time Per Output Token / Inter-Token Latency) | 첫 token 이후 token 간 간격 | decode loop | HBM bandwidth, KV read, sampling overhead, TP collective |

추론 방식:
- **첫 token이 느림** → TTFT 문제 → prefill 쪽(prompt 길이, prefill latency, prefix cache hit) 확인
- **token이 뚝뚝 끊김** → TPOT/ITL 문제 → decode 쪽(decode batch, HBM 대역폭, KV bytes/request) 확인

> 전체 latency ≈ TTFT + (output length × TPOT)

---

## 3. KV cache 공식에서 context length와 batch의 영향

```
KV bytes = layers × batch × seq_len × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
```

이 공식은 **선형 관계**가 핵심이다.

- `seq_len`(context length)이 4배 → KV cache도 **4배**
- `batch`가 2배 → KV cache도 **2배**
- `num_kv_heads`가 줄면 → KV cache도 줄어듦

즉 **context length와 batch는 곱(multiplicative)으로 작용**한다. 예를 들어 128k context에서 batch를 4로 올리면 KV cache는 약 4배로 커진다. 이 때문에 long context + 높은 concurrency 조합은 HBM을 가장 먼저 채우는 OOM/concurrency 제약 요인이 된다.

> weight가 GPU에 들어가는가만 보면 안 된다. weight + KV cache + activation/workspace + fragmentation 여유가 **모두** HBM에 들어가야 한다.

---

## 4. GQA/MQA가 decode와 KV cache에 유리한 이유

KV cache 크기는 `num_kv_heads`에 비례한다. GQA/MQA는 **KV head 수를 줄이는** 기법이다.

| 방식 | 구조 | KV cache 영향 |
|---|---|---|
| MHA | query head 수 = KV head 수 | KV가 큼 |
| GQA | 여러 query head가 더 적은 KV head를 공유 | KV가 줄어듦 |
| MQA | KV head를 거의 1개로 공유 | KV가 크게 줄어듦 |

decode에 유리한 두 가지 이유:
1. **KV 저장량(HBM capacity) 감소** → 더 긴 context, 더 많은 동시 요청 수용 가능
2. **KV read bytes(HBM bandwidth) 감소** → decode는 bandwidth-bound이므로 매 step 읽는 bytes가 줄면 TPOT가 직접 개선됨

decode 병목의 두 축(capacity, bandwidth)을 **동시에** 완화하기 때문에 중요하다.

---

## 5. TP, PP, DP를 모델을 나누는 축으로 구분

모델을 어떤 축으로 나누는가로 보면 명확하다.

| 방식 | 나누는 축 | 구조 | 주 병목 자원 |
|---|---|---|---|
| **TP** (Tensor Parallelism) | layer **내부** tensor/weight 차원 | 각 레이어 안의 큰 weight/activation을 여러 GPU가 나눠 계산, layer마다 all-reduce/all-gather | NVLink/NVSwitch (또는 cross-node IB/RDMA) — layer마다 동기화 |
| **PP** (Pipeline Parallelism) | layer **깊이**(stage) | 전체 layer stack을 stage 단위로 분할, stage 간 activation handoff | inter-stage link, pipeline bubble |
| **DP** (Data Parallelism) | **request/replica** | full model replica를 복제하고 요청을 나눠 보냄 | replica당 HBM capacity (weight+KV 복제), load balancer queue |

선택 규칙 요약:
- 7B/8B가 GPU 한 장에 들어감 → single GPU 또는 **DP**
- 70B가 한 GPU에 안 들어감 → 노드 내부 **TP=2/4**
- 405B+ → **TP+PP** + quantization + multi-node
- 핵심 룰: **TP는 가능하면 같은 NVLink/NVSwitch domain 안에 묶는다** (layer마다 동기화가 들어가므로)

---

## 6. CPU/NVMe offload가 왜 자주 PCIe/NVMe-bound가 되는가

CPU DRAM과 NVMe는 HBM보다 **크지만 훨씬 멀다.**

```
GPU HBM <-> PCIe <-> CPU DRAM <-> NVMe
```

decode는 매 step마다 weight와 KV를 반복해서 읽는데(hot path), 이 데이터가 HBM 밖(CPU DRAM/NVMe)에 있으면 **매번 PCIe(또는 NVMe)를 건너야** 한다. PCIe/NVMe 대역폭은 HBM보다 훨씬 낮으므로 이 경로가 즉시 병목이 된다.

따라서 offload는:
- hot KV나 매 layer weight를 거기 두면 → PCIe/NVMe-bound로 추락
- **cold KV, reusable prefix, capacity overflow** 같은 자주 안쓰는 cold 데이터 보조 용도로만 쓰는 게 안전


---

## 7. PD disaggregation에서 KV transfer가 왜 TTFT handoff와 p99 병목이 되는가

PD disaggregation은 prefill pool과 decode pool을 **분리**한다(자원 경쟁 완화, 독립 sizing 가능). 대신 새 문제가 생긴다: **prefill이 만든 KV를 decode 쪽으로 옮겨야 한다.**

이 KV transfer가 병목이 되는 구조:
- KV transfer는 **prefill 완료와 첫 decode 사이**에 끼므로 그대로 **TTFT handoff** 시간으로 들어감
- transfer 시간은 **KV 크기 ∝ context length**에 비례 → long context일수록 치명적
- RDMA/NIC/PCIe congestion이 끼면 일부 요청만 느려져 p99(tail)를 흔듦

그래서 PD를 도입했는데 더 느려졌다면 KV transfer 경로(RDMA, NIC, PCIe, KV layout 변환, congestion)부터 본다.

---

## 한 줄 요약

| 질문 | 핵심 |
|---|---|
| Prefill vs Decode 병목 | compute-bound vs HBM bandwidth-bound (arithmetic intensity 차이) |
| TTFT vs TPOT | prefill/queue 구간 vs decode loop 구간 |
| KV 공식 | context와 batch가 **곱**으로 KV를 선형 증가 |
| GQA/MQA | KV head 축소 → capacity·bandwidth 동시 완화 |
| TP/PP/DP | layer 내부 / layer 깊이 / request·replica 축 |
| offload 병목 | hot path가 PCIe/NVMe를 반복 통과 → 즉시 bound |
| PD KV transfer | prefill→decode handoff에 끼어 TTFT·p99 악화 |
