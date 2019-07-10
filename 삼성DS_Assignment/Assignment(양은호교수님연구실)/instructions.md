<p style="text-align: center;font-size:40px;font-weight:bold;">
Monte Carlo Method
</p>
담당조교: 양준용(laoconeth@kaist.ac.kr), 정인교(jik0730@gmail.com)

**과제 요약: Monte Carlo 방법을 사용하여 negative binomial distribution의 mean을 구해봅니다.**

### Negative Binomial Distribution

Negative binomial distribution $NB(r,p)$은 다음과 같이 정의됩니다. 먼저, 동전이 하나 있습니다. 이 동전은 $p$의 확률로 앞면(1), $1-p$의 확률로 뒷면(0)이 나오는 동전입니다. 이 동전을 계속 튕깁니다. 동전의 뒷면이 $r$번 나오면, 반복을 종료하고, 현재까지 동전을 총 몇번 튕겻는지를 반환합니다. 즉, $NB(3, 0.5)​$ 분포의 경우, 동전을 여러번 튕겨서 (1,1,1,0,1,1,1,0,0) 이렇게 0이 3회 등장하게 되면 종료되고, 샘플된 값은 9가 되는 것입니다. 자세한 것은 위키피디아 엔트리 <https://en.wikipedia.org/wiki/Negative_binomial_distribution>를 참조하시면 됩니다.

### Monte Carlo Integration

Monte carlo integration은 적분 수식을 직접 계산하는 대신, 샘플링을 통하여 적분값의 근사값을 구하는 방법입니다. 어떠한 확률분포의 mean은 expectation $E[x]$로 주어집니다. Monte Carlo integration을 사용하여, 다음과 같이 어떠한 $E[x]$의 근사값을 구할 수 있습니다.
$$
E[x] = \int{}{} xf(x)dx \approx \frac{1}{N} \sum_{n}^{N} x_{n}
$$

위 섹션에서 설명된 샘플링 절차를 거치면 $NB(r,p)​$ 분포로부터 샘플을 얻을 수 있습니다. 분포로부터 샘플을 얻을 수 있으므로, Monte Carlo 방법을 사용할 수 있습니다. 

### Todo

Monte Carlo를 사용하여 확률변수 $x \sim NB(r,p)$ 의 평균(mean)인 $E\left[x\right]$를 구하는 것이 목표입니다. 동봉된 skeleton code의 `sample_nb(r, p)` 와 `montecarlo(N)` 함수를 작성하시면 됩니다.

### Submission

montecarlo.py 파일을 제출하시면 됩니다.