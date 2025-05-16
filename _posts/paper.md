## absmax 양자화

input Tensor : $x \in \mathbb{R}$

Absolute Maximum: $x_\max = \max(|x|)$

Scaling Factor: $s$

Quantized Value: $q \in [-127, 127]$

Dequantized Value: $\hat{x}$

Integer Range: $[q_{\min}, q_{\max}]$
$$
x_\max = \max(|x|)
$$

$$
s = 
\frac{x_\max}
{q_\max}
$$

$$
q_i = \text{clip}(\lfloor \frac{x_i}{s} \rceil, q_\min, q_\max)
$$

$$
\hat{x_i} = s \cdot q_i
$$

## 소프트맥스

$$
\text{softmax}(x_i) = 
\frac{e^{x_i}}
{\sum_{j=1}^d e^{x_j} }
$$

## 자연 지수 함수 

1. 밑 변환 공식 

$$
a^x = b^{x \cdot \log_b{a}}
$$

2. 자연 지수 함수의 밑 변환 

$$
e^x = 2^{x \cdot \ln2} = 2^{x \cdot 0.683147}
$$

## 정수 기반 소프트맥스 

$$
\text{softmax}(x_i) = 
\frac{e^{x_i}}
{\sum_j^d e^{x_j} } = \frac{e^{q_i \cdot s}}
{\sum_j^d e^{q_i \cdot s}}
$$

## 안전한 소프트맥스 (Safe-Softmax)

1. 오버플로우 방지: $x_i - x_\max \leq 0 $ 
2. 언더플로우 방지: $\max(e^{x_i - x_\max}) = 1$

$$
\text{softmax}(x_i) = 
\frac{e^{x_i}}
{\sum_{j=1}^d e^{x_j} } = 
\frac{e^{x_i - x_\max}}
{\sum_{j=1}^d e^{x_j - x_\max}}
$$

## 정수 기반 안전한 소프트맥스 

$$
\text{Softmax}(x_i) = 
\frac{e^{x_i - x_\max}}
{\sum_{j=1}^d e^{x_j - x_\max}} \approx 
\frac{e^{s \cdot q_i  - s \cdot q_\max}}
{\sum_{j=1}^d e^{s \cdot q_j - s \cdot q_\max}} = 
\frac{e^{s \cdot (q_i - q_\max)}}
{\sum_{j=1}^d e^{s \cdot (q_j - q_\max)}} = 
\frac{e^{s \cdot \delta_{q_i}}}
{\sum_{j=1}^d e^{s \cdot \delta_{q_i}}}
$$

## 정수 기반 자연 지수 함수

1. 자연 지수 함수를 2의 지수함수로 변환 

$$
e^{s \cdot \delta_{q_i}} = 2^{s \cdot \delta_{q_i} \cdot \ln2} = 2^{(s\cdot \ln2) \cdot \delta_{q_i}} = 2^{\alpha \cdot \delta_{q_i}}
$$

2. 몫과 나머지로 분리 ($\delta_{q_i} \leq 0$ $ 이며 s \cdot \delta_{q_i} \leq 0$ 이라 $\alpha \cdot \delta_{q_i} \leq 0 $ 이다.)
   $$
   \begin{align*}
   \beta &= \lfloor 1 / \alpha \rceil \\
   k &= \lfloor \delta_{q_i} / (-\beta) \rfloor \\
   r &=  -(\delta_{q_i} - k \cdot(-\beta))
   \end{align*}
   $$
   
   
   

$$
2^{\alpha \cdot \delta_{q_i}} =2^{(-k) + \alpha \cdot (-r)} = 2^{\alpha \cdot (-r)} \gg k
$$

3. $\alpha \cdot (-r) \in (-1, 0]$ 임으로 일차 다항식 근사 가능 
   
   - 테일러 전개를 통해 근사를 하면 다음과 같다. 
   
   $$
   0.497646 x  + 0.970170, \quad \forall x \in (-1, 0]
   $$
   
   ![image-20250515232845900](/Users/etri/Desktop/blog/images/2025-05-15-paper/image-20250515232845900.png)
   
   - 이를 2의 거듭제곱 연산으로 근사하면 다음과 같다. 
   
   $$
   \quad 2^{\alpha \cdot q_i} \approx \frac{\alpha \cdot q_i}{2} + \frac{31}{32}, \quad \forall \alpha \cdot q_i \in (-1, 0]
   $$
   
   
   ![image-20250515231336696](/Users/etri/Desktop/blog/images/2025-05-15-paper/image-20250515231336696.png)
   
   - $2^x$를 기준으로 SQNR을 구하면 다음과 같다.

|      | x/2 +1 | x/2 + 31/32 | 0.497646 * x + 0.970170 |
| -- | -- | -- | -- |
| SQNR | 27.39  | 34.99 | 35.19 |

4. $2^{\alpha \cdot (-r)}$ 를 정수 기반 연산으로 구현한다. 

$$
\begin{align*}
2^{\alpha \cdot (-r)} &\approx \alpha \cdot (-r) / 2 + 31/32  \\
&= \alpha \cdot (((-r) \gg 1) + (\beta - \beta \gg 5))
\end{align*}
$$

5. 정리

$$
\begin{align*}
e^{s \cdot \delta_{q_i}} &= 2^{s \cdot \delta_{q_i} \cdot \ln2} \\
&= 2^{(s\cdot \ln2) \cdot \delta_{q_i}} \\
&= 2^{\alpha \cdot \delta_{q_i}} \\ 
&= 2^{\alpha \cdot (-r)} \gg k \\
&= \alpha \cdot (((-r) \gg 1) + (\beta - (\beta \gg 5))) \gg k
\end{align*}
$$

## 어텐션

$$
\begin{align*}
S &= QK^\top \\
P &= \text{softmax(S)} \\
O &= AV

\end{align*}
$$

## 플래시 어텐션

0. define 
   - $b$ : block size, (i.e. $\text{BLOCK\_N}$)
   - $i$ : block index, $1 \leq i < \text{N}\_\text{CTX} / \text{BLOCK\_N}$

1. $x_i$ : dot products between query and key 

$$
\boldsymbol{x_i} = \boldsymbol{Q}[k, :]\boldsymbol{K}^\top[:, (i-1)b:ib]
$$



2. $m_i^{(\text{local})}$ : maximum value in $i$-th block

$$
m_i^{(\text{local})} = \max_{j=1}^{b} \boldsymbol{x_i}[j]
$$

3. $m_i$ : maximum value up to the $i$-th block

$$
\begin{align*}
m_i &= \text{max}(m_{i-1},\: m_{i}^{\text{(local)}})
\end{align*}
$$

4. $p_j$ : unnormalized attention weight

$$
p_j = e^{(\boldsymbol{x_i}[j] - m_i)}
$$

5. $l_i^{(\text{local})}$ : local normalization factor

$$
l_i^{(\text{local})} = \sum_{j=1}^b p_j
$$



6. $\alpha_i$ : rescale old sum to new max 

$$
\alpha_i = e^{m_{i-1} - m_{i}}
$$

6. $l_i$ : cumulative normalization factor

$$
\begin{align*}
l_i
&= l_{i-1} \alpha_i + l_i^{(\text{local})}
\end{align*}
$$

8. $\boldsymbol{acc}_i^{(\text{local})}$ : local weighted value

$$
\boldsymbol{acc}_i^{(\text{local})} = \sum_{j=1}^b p_j \boldsymbol{V}[(i-1)b + j, :]
$$



9. $\boldsymbol{acc}_i$ : accumulated weighted value

$$
\boldsymbol{acc}_i = \boldsymbol{acc}_{i-1} \alpha_i + \boldsymbol{acc}_i^{(\text{local})}
$$

10. $\boldsymbol{O}[k,:]$ : attention output

$$
\boldsymbol{O}[k, :] = \frac{\boldsymbol{acc}_{\text{N\_CTX/BLOCK\_N}}}{l_\text{N\_CTX/BLOCK\_N}}
$$

## 재양자화 

- $s_{\text{in}}$: input scale 
- $s_{\text{out}}$: output scale 
- $s_{\text{req}}$ : requantization rescaling factor

$$
\begin{align*}
x^{(\text{int8})} &= \text{req}(x^{(\text{int32})}, s_\text{in},s_\text{out} ) \\
&= \text{req}(x^{(\text{int32})}, s_{\text{req}}) \\
&= \lfloor x^{(\text{int32})} \cdot s_{\text{req}} \rceil 
\end{align*}
$$

- $n$ : scaling shift
- $m_{\text{req}}^{{(\text{int32})}}$ : requantization multiplier 

$$
m_{\text{req}} = s_{\text{req}} \cdot 2^{n} 
$$

$$
\begin{align*}
x^{(\text{int8})} &= \text{req}(x^{(\text{int32})}, m_{\text{req}}, n) \\
&= (x^{(\text{int32})} \cdot m_{\text{req}} )\gg n 
\end{align*}
$$

### 쉬프트 기반 반올림

- 쉬프트 기반 반올림 적용 X 

$$
\left \lfloor \frac{x^{(\text{int32})}}{2^n} \right \rfloor = x^{(\text{int32})} \gg n
$$

- 쉬프트 기반 반올림 적용 O 

$$
\left \lfloor \frac{x^{(\text{int32})}}{2^n} \right \rceil = (x^{(\text{int32})} + 2^{n-1}) \gg n
$$



## 정수 기반 플래시 어텐션

0. define 
    - $b$ : block size, (i.e. $\text{BLOCK\_N}$)
    - $i$ : block index, $1 \leq i < \text{N}\_\text{CTX} / \text{BLOCK\_N}$
    - $\boldsymbol{Q} \approx s_Q \cdot \boldsymbol{Q}^{(\text{int8})}$  
    - $\boldsymbol{K} \approx s_K \cdot \boldsymbol{K}^{(\text{int8})}$  
    - $\boldsymbol{V} \approx s_V \cdot \boldsymbol{V}^{(\text{int8})}$ 
    - $\boldsymbol{O} \approx s_O \cdot \boldsymbol{O}^{(\text{int8})}$   
1. $\boldsymbol{x}_i^{(\text{int32})}$ 

$$
\boldsymbol{x}_i^{(\text{int32})} = \boldsymbol{Q}^{(\text{int8})}[k, :] \cdot \left( \boldsymbol{K}^{(\text{int8})} \right)^\top[:, (i-1)b : ib]
$$

2. $s_x$ : scaling factor of $\boldsymbol{x}_i^{(\text{int32})}$

$$
s_x = s_Q \cdot s_K
$$



3. $m_i^{(\text{local},\, \text{int32})}$

$$
m_i^{(\text{local},\, \text{int32})} = \max_{j=1}^{b} \boldsymbol{x}_i^{(\text{int32})}[j]
$$

4. $p_j^{(\text{int32})}$

$$
p_j^{(\text{int32})} = \text{i-exp}(\boldsymbol{x}_i[j]^{(\text{int32})} - m_i^{(\text{int32})}, s_x \cdot \ln2)
$$

5. $m_i^{(\text{int32})}$ 

$$
\begin{align*}
m_i^{(\text{int32})} &= \text{max}(m_{i-1}^{(\text{int32})},\: m_{i}^{\text{(local, int32)}})
\end{align*}
$$

6. $s_p^{(\text{in})}$ : scaling fator of $p_j^{(\text{int32})}$

$$
s_p^{(\text{in})} = s_x \cdot \ln 2
$$

7. $s_p^{(\text{out})}$ : scaling factor of $p_j^{(\text{int8})}$

$$
p_j \approx p_j^{(\text{int32})} \cdot s_p^{(\text{in})}, \quad \forall j : p_j \leq 0
$$

$$
p_j^{(\text{int8})} = \lfloor p_j^{} \cdot 127 \rceil , \quad \forall j: p_j^{(\text{int8})} \leq 127
$$

$$
s_p^{(\text{out})} = \frac{1}{127}
$$



9. $s_p^{(\text{req})}$ : requantization scaling factor

$$
s_p^{(\text{req})} = \frac{s_p^{(\text{in})}}{s_p^{(\text{out})}}
$$

7. $p_j^{(\text{int8})}$ : requantization of $p_j^{(\text{int32})}$

$$
p_j^{(\text{int8})} = \text{req}(p_j^{(\text{int32})},s_p^{(\text{req})})
$$

8. $l_i^{(\text{local, int32})}$ 

$$
l_i^{(\text{local, int32})}=\sum_{j=1}^b p_j^{(\text{int8})}
$$

8. $\alpha_i^{(\text{int32})}$ 

$$
\alpha_i^{(\text{int32})} = \text{i-exp}(m_{i-1}-m_i, \: s_x \cdot \ln2)
$$

11. $s_\alpha$ : scaling factor of $\alpha_i^{(\text{int32})}$

$$
s_\alpha = s_x \cdot \ln2
$$

12. $s_\beta^{(\text{int32})}$ : inverse scale as integer 

$$
s_\beta^{(\text{int32})} = \left\lfloor \frac{1}{s_\alpha} \right\rceil
$$

13. $l_i^{(\text{int32})}$

$$
l_i^{(\text{int32})} = \left \lfloor \frac{l_{i-1}^{(\text{int32})} \cdot \alpha_i^{(\text{int32})}}
{s_\beta^{(\text{int32})}} \right \rfloor
+ l_i^{(\text{local, int32})}
$$

14. $\boldsymbol{acc}_i^{(\text{local, int32})}$ 

$$
\boldsymbol{acc}_i^{(\text{local, int32})} = \sum_{j=1}^b p_j^{\text{(int8)}} \boldsymbol{V}^{\text{(int8)}}[(i-1)b + j, :]
$$

15. $\boldsymbol{acc}_i^{(\text{int32})}$

$$
\boldsymbol{acc}_i^{(\text{int32})} = \left \lfloor \frac{\boldsymbol{acc}_{i-1}^{(\text{int32})} \alpha_i^{(\text{int32})}}
{s_\beta^{(\text{int32})}} \right \rfloor + \boldsymbol{acc}_i^{(\text{local, int32})}
$$

16. $\boldsymbol{O}[k,:]$ : attention output

$$
\boldsymbol{O}[k, :] = \frac{\boldsymbol{acc}_{\text{final}}^{(\text{int32})}}{l_\text{final}^{(\text{int32})}} \cdot 
s_p^{(\text{out})} \cdot s_V
$$

## $l_i$ 에 대한 전개 

1. $l_0$

$$
l_0 = \sum_{j=1}^b e^{(\boldsymbol{x_0}[j] - m_0)}
$$

2. $l_1$

$$
\begin{align*}
l_1 &= \sum_{j=1}^b e^{(\boldsymbol{x_0}[j] - m_0)} \cdot e^{m_0 - m_1} \\
&+ \sum_{j=1}^b e^{(\boldsymbol{x_1}[j] - m_1)}
\end{align*}
$$

3. $l_2$

$$
\begin{align*}
l_1 &= \sum_{j=1}^b e^{(\boldsymbol{x_0}[j] - m_0)} \cdot e^{m_0 - m_1} \cdot e^{m_1 - m_2}\\
&+ \sum_{j=1}^b e^{(\boldsymbol{x_1}[j] - m_1)} \cdot e^{m_1 - m_2} \\
&+ \sum_{j=1}^b e^{(\boldsymbol{x_2}[j] - m_2)}
\end{align*}
$$

## $l_i^{(\text{int32})}$ 에 대한 전개

- $s_\alpha$ 와 $s_\beta^{(\text{int32})}$ 는 역수 관계 

1. $l_{1}^{(\text{int32})}$

$$
l_0^{(\text{int32})} = \sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_1}[j]^{\text{(int32)}} - m_1^{(\text{int32})})}
$$

2. $l_2^{(\text{int32})}$

$$
\begin{align*}
l_2^{(\text{int32})} &= \frac{\sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_1}[j]^{\text{(int32)}} - m_1^{(\text{int32})})}
\cdot \text{i-exp}(m_1-m_2, s_\alpha)}{s_\beta^{(\text{int32})}} \\
&+ \sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_2}[j]^{\text{(int32)}} - m_2^{(\text{int32})})}
\end{align*}
$$

3. $l_3^{(\text{int32})}$

$$
\begin{align*}
l_3^{(\text{int32})} &= \frac{\sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_1}[j]^{\text{(int32)}} - m_1^{(\text{int32})})}
\cdot \text{i-exp}(m_1-m_2, s_\alpha) \cdot \text{i-exp}(m_2 - m_3 , s_\alpha)}{s_\beta^{(\text{int32})} s_\beta^{(\text{int32})}} \\
&+ \frac{\sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_2}[j]^{\text{(int32)}} - m_2^{(\text{int32})})}\cdot \text
{i-exp}(m_2 - m_3, s_\alpha)}{s_\beta^{(\text{int32})}} \\
&+ \sum_{j=1}^b \text{i-exp}{(\boldsymbol{x_3}[j]^{\text{(int32)}} - m_3^{(\text{int32})})}
\end{align*}
$$

## $\boldsymbol{acc}_i^{(\text{int32})}$에 대한 전개 

- $\alpha_i^{(\text{int32})} = \text{i-exp}(m_{i-1}-m_i, \: s_\alpha)$ 
- $s_\beta = \left \lfloor \frac{1}{s_\alpha} \right \rceil$

1. $\boldsymbol{acc}_1^{(\text{int32})}$

$$
\boldsymbol{acc}_1^{(\text{int32})} = \sum_{j=1}^b p_j^{\text{(local, int8)}} \boldsymbol{V}^{\text{(int8)}}[j, :]
$$

2. $\boldsymbol{acc}_2^{(\text{int32})}$

$$
\begin{align*}

\boldsymbol{acc}_2^{(\text{int32})} &= \frac{\sum_{j=1}^b p_j^{\text{(int8)}} \boldsymbol{V}^{\text{(int8)}}[j, :] 
\cdot \alpha_2^{(\text{int32})}}{s_\beta} \\ 
&+ \sum_{j=b+1}^b p_j^{\text{(int8)}} \boldsymbol{V}^{\text{(int8)}}[j, :]

\end{align*}
$$

