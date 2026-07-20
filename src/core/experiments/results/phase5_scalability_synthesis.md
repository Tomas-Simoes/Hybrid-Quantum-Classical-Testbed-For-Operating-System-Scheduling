# Síntese corrigida da escalabilidade — N=10 a N=1000

Configuração fixa: K=2, mixer XY, p=2, 100 passos, `qubit_max=8`,
`top_k=10`, pesos uniformes aleatórios. O piloto N=10 e o sweep N=20–200
usaram 20/12/8 repetições por patamar; o stress N=300–1000 usou três seeds.

| N | Runs | Sub-QUBOs | Imbalance médio | Imbalance / W total | Referência média | Tempo médio |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 3 | 0.086378 | 1.608695% | 0.005424 | 4.9 s |
| 20 | 20 | 5 | 0.053202 | 0.481645% | 0.000006 | 17.3 s |
| 25 | 20 | 7 | 0.019723 | 0.146499% | 0.000014 | 16.6 s |
| 30 | 20 | 8 | 0.031892 | 0.208845% | 0.000008 | 19.3 s |
| 40 | 20 | 10 | 0.014212 | 0.070449% | 0.000008 | 25.7 s |
| 50 | 20 | 13 | 0.014861 | 0.057691% | 0.000006 | 33.1 s |
| 65 | 12 | 17 | 0.015765 | 0.047720% | 0.000005 | 43.3 s |
| 80 | 12 | 20 | 0.011203 | 0.028148% | 0.000005 | 53.4 s |
| 100 | 12 | 25 | 0.008121 | 0.016461% | 0.000003 | 65.2 s |
| 130 | 8 | 33 | 0.007540 | 0.011080% | 0.000005 | 85.9 s |
| 160 | 8 | 40 | 0.008459 | 0.009987% | 0.000005 | 104.9 s |
| 200 | 8 | 50 | 0.004122 | 0.003959% | 0.000007 | 134.4 s |
| 300 | 3 | 75 | 0.001456 | 0.000960% | 0.000008 | 157.0 s |
| 500 | 3 | 125 | 0.000905 | 0.000362% | 0.000012 | 324.2 s |
| 750 | 3 | 188 | 0.001269 | 0.000334% | 0.000019 | 476.5 s |
| 1000 | 3 | 250 | 0.001013 | 0.000198% | 0.000029 | 672.5 s |

## Conclusões

- As 192/192 execuções foram bem-sucedidas e globalmente viáveis; todos os
  sub-QUBOs também foram viáveis.
- Nenhuma solução da pipeline foi globalmente ótima. N=10 e N=20 foram
  decididos por referência exata; em N=20–1000 o pós-check MILP encontrou
  uma atribuição válida melhor em cada execução, provando não otimalidade sem
  precisar de conhecer o ótimo.
- Apesar disso, não há degradação operacional. O imbalance médio relativo à
  carga total caiu de 1.61% em N=10 para 0.000198% em N=1000.
- O tempo QAOA cresce aproximadamente com o número de sub-QUBOs. N=1000
  demorou em média 11.2 minutos por run, dos quais 8.2 minutos foram QAOA.
- Acima de N=20, a referência principal do sweep é simulated annealing e não
  um certificado ótimo. O MILP é usado como pós-check: os seus testemunhos
  melhores provam subotimalidade, mas o limite curto não certifica o ótimo.
- Estes resultados demonstram escalabilidade da decomposição/orquestração com
  blocos fixos de oito qubits; não demonstram escalabilidade do tamanho do
  circuito quântico.
