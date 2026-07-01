# Protocolo

Foram importadas as seguintes bibliotecas necessárias para o desenvolvimento do projeto: 
- Pandas para a manipulação do dataset,
- Matplotlib para a geração de gráficos e Scikit-learn para o treinamento e avaliação do modelo de Machine Learning.

Após o carregamento do arquivo no formato CSV, foi feita uma análise exploratória inicial por meio das seguintes funções:

- head(): para a visualização das linhas iniciais e compreensão da estrutura dos dados;
- info(): para a verificação dos tipos de dados de cada coluna;
- describe(): para a obtenção do resumo estatístico das variáveis numéricas.

Em seguida, foi feita uma análise estatística descritiva detalhada que avaliou média, mediana, desvio padrão, valores mínimos e máximos.

A varíavel predefinida para a tarefa de classificação foi a coluna Alta_Ansiedade. O critério estabelecido determinou que registros com pontuação de ansiedade maior ou igual a 7 seriam classificados como "alta ansiedade", enquanto os demais seriam categorizados como "baixa ansiedade". A coluna original contendo a pontuação contínua de ansiedade foi removida do dataset. Isso evitou o vazamento de dados, impedindo que o modelo tivesse acesso direto à resposta correta durante o treinamento e as variáveis categóricas foram convertidas em valores numéricos utilizando a técnica de LabelEncoder.

Os dados foram divididos na proporção de 70% para o conjunto de treinamento e 30% para o conjunto de teste, garantindo uma base de validação com dados não observados pelo modelo durante o ajuste. O algoritmo selecionado para o problema de classificação foi o Random Forest, porque é um algoritmo bastante utilizado em problemas de classificação e costuma apresentar bons resultados nesse tipo de dataset. Após o treinamento com o conjunto correspondente, o modelo foi submetido ao conjunto de teste para a geração das previsões.

A avaliação do desempenho ocorreu por meio das seguintes métricas:

- Acurácia, Precisão, Recall e F1-Score;
- Matriz de Confusão,
- Relatório de Classificação.
