---
layout: single
title:  "텐서 코어와 쿠다 코어"
categories: "코딩"
toc: true
typora-root-url: .
---

## Tensor Core 

Tensor Core는 NVIDIA GPU의 SM(Streaming Multiprocessor) 내부에 통합된 전용 MMA(Matrix Multiply-Accumulate) 유닛으로, 딥러닝의 핵심 연산인 대규모 행렬 곱을 극도로 병렬화하여 처리량을 비약적으로 높여줍니다. 

### 세대별 지원

1. Volta (sm_70, Tesla V100)
   - 첫 도입(2017년): FP16 입력 $\rightarrow$ FP32 누산(Accumulate) MMA 연산을 지원합니다. 
2. Tuning (sm_75, GeForce RTX 20 시리즈)
   - FP16 이외에 INT8, INT4 정수 연산 모드가 추가되었습니다. 
3. Ampere (sm_80, A100)
   - BFloat16(BF16) 및 TensorFloat-32(TF32) 지원합니다.
   - FP64 IEEE-compliant MMA 지원 (FP64 입력 $\rightarrow$ FP64 누산)
   - INT8, INT4 희소 연산 지원합니다.
   - 희소성(sparsity) 가속 지원합니다. 
4. Hopper (sm_90, H100)
   - FP8 (E4M3, E5M2) 입력 형식 네이티브 지원합니다. 
     - E4M3: 최대 값: $448$, 최소값: $0.015625$
     - E5M2: 최대 값: $57344$, 최소 값: $6.1035 \times 10^{-5}$
   - 비동기(`mma_async`) 연산, 공유 메모리에서 연산 유닛이 직접 데이터 참조 가능합니다. 
5. Blackwell
   - Micro‑tensor scaled FP4, FP6, FP8 신규 부동소수점 포맷 $\rightarrow$ 양자화의 float 버전(`fp32_val = fpX_val x scale`)

### gemm_fp16

```c++
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C"
__global__ void wmma_gemm_fp16(const half *A, const half *B, float *C, int K, int N) {
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x; 

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const half* tileA = A + tileRow * WMMA_M * K + k0; 
        const half* tileB = B + k0 * N + tileCol * WMMA_N;

        wmma::load_matrix_sync(aFrag, tileA, K);
        wmma::load_matrix_sync(bFrag, tileB, N);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    float *tileC = C + tileRow * WMMA_M * N + tileCol * WMMA_N;
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);
}
```

### gemm_bf16

```c++
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C"
__global__ void wmma_gemm_bf16(const nv_bfloat16 *A, const nv_bfloat16 *B, float *C, int K, int N) {
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const nv_bfloat16* tileA = A + tileRow * WMMA_M * K + k0;
        const nv_bfloat16* tileB = B + k0 * N + tileCol * WMMA_N;

        wmma::load_matrix_sync(aFrag, tileA, K);
        wmma::load_matrix_sync(bFrag, tileB, N);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);  // Supported on sm_80+
    }

    float *tileC = C + tileRow * WMMA_M * N + tileCol * WMMA_N;
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);
}
```



## CUDA Core 

NVIDIA GPU의 가장 기본적인 연산 유닛입니다.

### Dot Product of 4-element 8-bit integers and Accumulate (`dp4a`)

네 개의 곱셈 + 합산을 하나의 명령어로 , 한 사이클에 실행 가능합니다. 

```C++
extern "C" __global__
void dp4a_kernel(const int* a, const int* b, int* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int acc = 0;
    int a_val = 0, b_val = 0;

    if (idx < N) {
        a_val = a[idx];
        b_val = b[idx];
    }

    asm volatile(
        "dp4a.s32.s32 %0, %1, %2, %3;" :
        "=r"(acc) : "r"(a_val), "r"(b_val), "r"(acc)
    );

    if (idx < N)
        out[idx] = acc;
}
```

**INT32 곱셈기 모델**

```c++
int mul_bitwise_int32(int32_t A, int32_t B) {
  
  	int32_t partial = 0;
  
    #pragma omp parallel for reduction(+:partial)
    for (int i = 0; i < 32; ++i) {
        if ((B >> i) & 1) {
            partial += (A << i);
        }
    }

    return partial;
}

```

**INT32 곱셈 예시**
$$
\begin{align*}
5  \times 3&= 15\\
0\text{b}0101 + 0\text{b}0101 \ll 1 &= 0\text{b}1111
\end{align*}
$$


**INT8 곱셈기 모델**

```c++
int32_t mul_bitwise_int8(int8_t a, int8_t b) {

  	int32_t partial = 0;
    int32_t A = (int32_t)a;
    int32_t B = (int32_t)b;

		#pragma omp parallel for reduction(+:partial)
    for (int i = 0; i < 8; ++i) {
        if ((B >> i) & 1) {
            partial += (A << i); 
        }
    }

    return partial;
}
```

**dp4a 모델** 

```C++
int32_t mulacc_dp4a(const int8_t a[4], const int8_t b[4], int32_t acc) {

		int32_t sum = 0;
  
		#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 4; ++i) {
        sum += mulacc_bitwise_int8(a[i], b[i]);
    }
    return acc + sum;
}
```









