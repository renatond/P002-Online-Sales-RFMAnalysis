# Online Sales & RFM Analysis

Esse repositório contém códigos da análise do portfólio de uma marca de Chocolates no Brasil.<br>
Todas as informações abaixo são reais e anonimizadas, preservando a propriedade da empresa.

## Objetivo
* Realizar a análise exploratória de dados de compras disponíveis no conjunto de dados.
* Realizar uma análise Cohort dos clientes.
* Identificar produtos com maior venda conjunta (cross-sell)
* Desenvolver uma segmentação de clientes com base no comportamento de compras.
* Desenvolver um [dashboard](https://case-chocolate.herokuapp.com/) online que possa ser acessado pelo CEO através de um celular ou computador.

---
## 1. Problema do Negócio
O modelo de negócios da marca consiste em vendas no varejo, tanto com lojas físicas, onde o cliente tem uma experiênci amais intimista, quanto no varejo online. O cientista de dados é responsável por desenvolver um dashboard online para que o CEO da empresa tenha uma visão geral das vendas através do site e do comportamento dos consumidores que optam por esta modalidade.<br>

<br>O [dashboard](https://case-chocolate.herokuapp.com/) deve conter:
   * Ticket Médio e Faturamento da Empresa
   * Uma visualização da distribuição mensal do faturamento.
   * Uma análise cohort dos clientes do site.
   * Identificação das combinações de produtos mais vendidos em conjunto.
   * Segmentação dos clientes.

## 2. Resultados
Faturamento: R$2,58Mi
Ticket Médio:R$191,05
Média de Retenção dos clientes: 3,13%
Quase 95% dos clientes que já compraram no site se tornam inativos.

## 3. Premissas
* Os dados disponíveis são de março a agosto de 2022.
* Tanto "fulfillment" quanto "marketplace" foram considerados como vendas do site.
* Categorização dos clientes conforme:
    * Inativo: Score < 3
    * Baixo Valor: Score 3 e 4
    * Médio Valor: Score 5 e 6
    * Alto Valor: Score 7 ou superior

* As variáveis originais do conjuto de dados são:<br>

Variável | Definição
------------ | -------------
|Origem | Origem da venda|
|ID Compra | Identificação única da compra|
|Data | Data da compra |
|Cliente | Identificação única do cliente|
|Quantidade SKU | Quantidade do produto na compra|
|ID_SKU | Identificação única do produto|
|Valor Unitário SKU | Valor unitário do produto|
|Valor Total SKU | Valor total de um produto na compra (Qtd x Valor Unitário)|
|Valor Total da Compra |Valor total de uma mesma compra|

## 4. Estratégia de Solução
1. Compreender o modelo de negócios
2. Compreender o problema de negócio
3. Coletar os Dados
4. Análise Exploratória dos Dados
5. Feature Engineering
6. Análise de Cohort
9. Análise RFM para segmentação dos clientes
10. Publicação do [Dashboard](https://case-chocolate.herokuapp.com/)
<br>

## 5. Insights
1. Clientes com comprar a primeira vez na loja em virtude da páscoa, tem menor retenção ao longo do ano.
2. Clientes que compraram pela primeira vez no mês que antecede a Páscoa tem maior retenção, com mais de 8% retornando para as compras de Páscoa.
3. Quase 95% dos Clientes se tornam inativos após a primeira compra.


## 6. Conclusão
O objetivo do projeto era fazer uma análise das vendas do site e do comportamento dos clientes. Com as features existentes no dataset era possível apenas a segmentação via uma análise RFM (Recência, Frequência, Monetário). 

## 7. Próximos Passos
* Ampliar a coleta de features para a base de dados de modo a permitir segmentação de produtos e clientes
* Get more address data to fill NAs.
* Confrontar resultados das vendas do site com resultados de vendas em lojas físcias.
<br>

---
## 8. Referencias:
